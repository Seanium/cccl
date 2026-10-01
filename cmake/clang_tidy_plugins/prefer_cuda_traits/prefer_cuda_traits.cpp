// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
// reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <clang-tidy/ClangTidy.h>
#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang/AST/QualTypeNames.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Lex/Lexer.h>
#include <clang/Tooling/Refactoring/Lookup.h>
#include <llvm/ADT/SmallVector.h>

namespace clang::tidy::cccl
{
namespace
{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

AST_MATCHER(Type, isDependentType) // NOLINT
{
  return Node.isDependentType();
}

// Clang 22's hasAnyTemplateArgumentLoc does not support UnresolvedLookupExpr.
AST_MATCHER(UnresolvedLookupExpr, hasDependentTypeArgument) // NOLINT
{
  auto&& arguments = Node.template_arguments();

  return std::any_of(arguments.begin(), arguments.end(), [](const auto& argument_loc) {
    const auto& argument = argument_loc.getArgument();
    return argument.getKind() == TemplateArgument::Type && argument.getAsType()->isDependentType();
  });
}

class PreferCUDATraitsCheck final : public ClangTidyCheck
{
  static constexpr StringRef TRAITS_OPTION{"Traits"};

  static constexpr StringRef DECLARATION_BIND{"declaration"};
  static constexpr StringRef USE_BIND{"use"};
  static constexpr StringRef CONTEXT_BIND{"context"};

  class Replacement final : public MatchFinder::MatchCallback
  {
  public:
    Replacement(ClangTidyCheck& check, std::string from, std::string to)
        : source_{std::move(from)}
        , target_{StringRef(to).starts_with("::") ? std::move(to) : "::" + to}
        , check_{&check}
    {}

    void run(const MatchFinder::MatchResult& result) override
    {
      auto&& nodes                  = result.Nodes;
      const auto* const declaration = nodes.getNodeAs<NamedDecl>(DECLARATION_BIND);

      if (const auto* const type = nodes.getNodeAs<TemplateSpecializationTypeLoc>(USE_BIND))
      {
        auto&& ctx = *result.Context;
        const auto original =
          TypeName::getFullyQualifiedName(type->getType().getDesugaredType(ctx), ctx, ctx.getPrintingPolicy());

        apply_(result, declaration, type->getQualifierLoc(), type->getTemplateNameLoc(), original);
        return;
      }

      if (const auto* const reference = nodes.getNodeAs<DeclRefExpr>(USE_BIND))
      {
        apply_(result,
               declaration,
               reference->getQualifierLoc(),
               reference->getLocation(),
               declaration->getQualifiedNameAsString());
        return;
      }

      if (const auto* const lookup = nodes.getNodeAs<UnresolvedLookupExpr>(USE_BIND))
      {
        apply_(
          result, declaration, lookup->getQualifierLoc(), lookup->getNameLoc(), declaration->getQualifiedNameAsString());
        return;
      }

      llvm::reportFatalUsageError("cccl-prefer-cuda-traits: unhandled node kind, this is a bug in the check");
    }

    [[nodiscard]] StringRef source() const
    {
      return source_;
    }

  private:
    [[nodiscard]] std::string make_replacement_(
      const MatchFinder::MatchResult& result,
      const clang::NamedDecl* declaration,
      NestedNameSpecifierLoc qualifier,
      SourceLocation name_loc) const
    {
      auto scope = qualifier.getNestedNameSpecifier();

      if (scope.isFullyQualified())
      {
        // Given the mapping cuda::foo -> cuda::std::foo:
        //
        // namespace cuda {
        //   ::cuda::foo<T> v;  // Must become ::cuda::std::foo<T> v;
        // }
        //
        // replaceNestedName() returns "std::foo" here, so use target_ to preserve the
        // leading ::.
        return target_;
      }

      if (!scope)
      {
        // Given the mapping cuda::foo -> cuda::std::foo:
        //
        // using                          cuda::foo;
        // foo<T>                         v;  // Must become cuda::std::foo<T> v;
        //
        // With an empty                  scope, replaceNestedName() returns "foo", which still names the
        // original trait. This is because replaceNestedName() assumes some other fixit will
        // update the using declaration:
        //
        // > other parts of this refactor will automatically fix using statements to point to the
        // > new class/function.
        //
        // https://github.com/llvm/llvm-project/blob/main/clang/lib/Tooling/Refactoring/Lookup.cpp
        //
        // We leave the using declaration unchanged, so supply the trait's namespace instead.
        if (const auto* const ns =
              dyn_cast<NamespaceDecl>(declaration->getDeclContext()->getEnclosingNamespaceContext()))
        {
          // namespace cuda {
          //   foo<T> v;  // Must become std::foo<T> v;
          // }
          //
          // Supply cuda:: without a leading :: so the helper can omit the enclosing namespace.
          scope = NestedNameSpecifier{*result.Context, ns, /*Prefix=*/std::nullopt};
        }
        // A trait declared at global scope has no NamespaceDecl, and replaceNestedName()
        // handles its empty scope directly.
      }

      const auto* use_context = result.Nodes.getNodeAs<Decl>(CONTEXT_BIND)->getDeclContext();

      return tooling::replaceNestedName(scope, name_loc, use_context, declaration, target_);
    }

    void apply_(const MatchFinder::MatchResult& result,
                const clang::NamedDecl* declaration,
                NestedNameSpecifierLoc qualifier,
                SourceLocation name_loc,
                StringRef original) const
    {
      const auto replacement = make_replacement_(result, declaration, qualifier, name_loc);
      auto diagnostic = check_->diag(name_loc, "use '%0' instead of '%1' for generic types") << replacement << original;

      // Allow macro fixes, but note that editing a macro definition affects every expansion:
      //
      // #define TRAIT cuda::std::foo // should be cuda::foo instead
      // TRAIT<T>   generic;
      // TRAIT<int> concrete; // oops, would also become cuda::foo<int>
      //
      // This is technically not in the spirit of this check, since we explicitly tried to
      // match only dependent names (the idea being that if you say cuda::foo<int> then you
      // really mean cuda::foo<int>), but this scenario is so rare that we can stomach the
      // potentially false positive here.
      const auto begin_loc = qualifier ? qualifier.getBeginLoc() : name_loc;

      // Given cuda::foo -> cuda::std::foo:
      //
      // cuda::foo<T>::value
      // ^-------^
      //   range
      const auto range = Lexer::makeFileCharRange(
        CharSourceRange::getTokenRange(begin_loc, name_loc), *result.SourceManager, result.Context->getLangOpts());

      // A partial macro expansion can have no corresponding file range:
      //
      // #define VALUE(T) cuda::foo_v<T>
      // static_assert(VALUE(T));
      //
      // The name range covers cuda::foo_v, but VALUE(T) expands to cuda::foo_v<T>. Replacing
      // the whole invocation would discard <T>, so makeFileCharRange returns an invalid range.
      if (range.isValid())
      {
        diagnostic << FixItHint::CreateReplacement(range, replacement);
      }
    }

    std::string source_{};
    std::string target_{};
    ClangTidyCheck* check_{};
  };

public:
  PreferCUDATraitsCheck(StringRef name, ClangTidyContext* context)
      : ClangTidyCheck{name, context}
      , traits_option_{Options.get(TRAITS_OPTION, "::cuda::std::is_trivially_copyable,::cuda::is_trivially_copyable;")}
  {
    llvm::SmallVector<StringRef> entries;

    traits_option_.split(entries, /*Separator=*/';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    replacements_.reserve(entries.size() * 3);
    for (auto&& entry : entries)
    {
      auto&& [from, to] = entry.split(',');

      from = from.trim();
      to   = to.trim();
      if (from.empty() || to.empty() || to.contains(','))
      {
        configurationDiag("invalid %0 entry '%1': expected from,to", DiagnosticIDs::Level::Error)
          << TRAITS_OPTION << entry;
        continue;
      }

      replacements_.emplace_back(*this, from.str(), to.str());
      if (!from.ends_with("_v") && !from.ends_with("_t"))
      {
        replacements_.emplace_back(*this, (from + "_v").str(), (to + "_v").str());
        replacements_.emplace_back(*this, (from + "_t").str(), (to + "_t").str());
      }
    }
  }

  void storeOptions(ClangTidyOptions::OptionMap& options) override
  {
    Options.store(options, TRAITS_OPTION, traits_option_);
  }

  void registerMatchers(MatchFinder* finder) override
  {
    // Only check generic types. For example:
    //
    // is_trivially_copyable<T>       // Matches.
    // is_trivially_copyable<T*>      // Matches.
    // is_trivially_copyable<int>     // Does not match.
    //
    // A concrete type does not have the same unknown CUDA special-member behavior.
    const auto generic_argument = hasAnyTemplateArgumentLoc(hasTypeLoc(loc(isDependentType())));

    for (auto&& replacement : replacements_)
    {
      const auto name        = hasName(replacement.source());
      const auto declaration = namedDecl(name).bind(DECLARATION_BIND);
      const auto use_context = hasAncestor(decl().bind(CONTEXT_BIND));

      // Given
      //
      // cuda::std::is_trivially_copyable<T> x;
      // static_assert(cuda::std::is_trivially_copyable<T>::value);
      // cuda::std::is_trivially_copyable<int> y;
      // cuda::is_trivially_copyable<T> z;
      //
      // Matches both occurrences of "cuda::std::is_trivially_copyable<T>".
      //
      // Does not match "cuda::std::is_trivially_copyable<int>" or
      // "cuda::is_trivially_copyable<T>".
      finder->addMatcher(
        traverse(
          // Ignore compiler-generated nodes, including copies from template instantiation.
          TK_IgnoreUnlessSpelledInSource,
          // Match the written type so the callback receives its source location.
          templateSpecializationTypeLoc(
            loc(templateSpecializationType(hasDeclaration(declaration))), generic_argument, use_context)
            .bind(USE_BIND)),
        &replacement);

      // Given
      //
      // static_assert(cuda::std::is_trivially_copyable_v<T>);
      // using cuda::std::is_trivially_copyable_v;
      // static_assert(is_trivially_copyable_v<T>);
      // static_assert(cuda::std::is_trivially_copyable_v<int>);
      // static_assert(cuda::is_trivially_copyable_v<T>);
      //
      // Matches "cuda::std::is_trivially_copyable_v<T>" and "is_trivially_copyable_v<T>" (but
      // not "cuda::std::is_trivially_copyable_v<int>" or "cuda::is_trivially_copyable_v<T>").
      //
      // The first matcher handles *types*, including "cuda::std::is_trivially_copyable<T>"
      // within "cuda::std::is_trivially_copyable<T>::value", but
      // "cuda::std::is_trivially_copyable_v<T>" references a variable template and therefore
      // requires a *expression* matcher.
      finder->addMatcher(
        traverse(
          // Ignore compiler-generated nodes, including copies from template instantiation.
          TK_IgnoreUnlessSpelledInSource,
          // Clang represents "cuda::std::is_trivially_copyable_v<int>" as a DeclRefExpr
          // pointing to the specialization. We skip this example because int is not a template parameter.
          declRefExpr(to(varDecl(name).bind(DECLARATION_BIND)), generic_argument, use_context).bind(USE_BIND)),
        &replacement);
      // Note, separate registrations let MatchFinder reject unrelated expression kinds before
      // it evaluates these matchers, so this ends up being faster than a single matcher with
      // anyOf().
      finder->addMatcher(
        traverse(
          // Ignore compiler-generated nodes, including copies from template instantiation.
          TK_IgnoreUnlessSpelledInSource,
          // A dependent lookup retains candidate declarations instead. For example:
          //
          // using cuda::std::is_trivially_copyable_v;
          // is_trivially_copyable_v<T>
          //
          // Matches "is_trivially_copyable_v<T>", but not the using declaration. Follow using
          // declarations to the original trait declaration.
          unresolvedLookupExpr(
            hasAnyDeclaration(namedDecl(hasUnderlyingDecl(declaration))), hasDependentTypeArgument(), use_context)
            .bind(USE_BIND)),
        &replacement);
    }
  }

private:
  StringRef traits_option_{};
  std::vector<Replacement> replacements_;
};

// ==========================================================================================

class PreferCUDATraitsCheckModule final : public ClangTidyModule
{
public:
  static constexpr StringRef CHECK_NAME{"cccl-prefer-cuda-traits"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<PreferCUDATraitsCheck>(CHECK_NAME);
  }
};

// Register the module using this statically initialized variable.
// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<PreferCUDATraitsCheckModule> _{
  PreferCUDATraitsCheckModule::CHECK_NAME, "Checks for use of cuda::std::meow which is superseded by cuda::meow"};
} // namespace
} // namespace clang::tidy::cccl
