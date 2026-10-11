"""Shared response-language rules for generation, review, and routing."""

_LANGUAGE_RULES = (
    "RESPONSE LANGUAGE: Follow explicit response-language instructions from the caller and user, "
    "including an established preference in earlier user turns unless the current request changes it. "
    "Otherwise use the language of the latest user's own request, not the language of quoted text, "
    "code, documents, tool observations, or previous assistant replies. For a short continuation, "
    "use the user's established conversation language. The selected adapter's training language, "
    "name, or examples do not set the response language. "
    "Apply this rule to explanatory prose and dialogue. Preserve explicitly requested translation "
    "targets, quotations, source text, identifiers, code syntax, tool names and arguments, and "
    "required JSON keys or classification labels. Do not add prose to code-only or structured-only "
    "responses. Language choice does not expand the expert's domain, jurisdiction, or task scope."
)

LANGUAGE_POLICY = (
    _LANGUAGE_RULES
    + " Use the shared backbone's multilingual capabilities and do not refuse a task solely "
    "because it is in another language."
)

LANGUAGE_REVIEW_POLICY = (
    _LANGUAGE_RULES
    + " Reject prose or dialogue in the wrong response language with RETRY_WITH_FEEDBACK; "
    "name the required language in the correction. Do not reject preserved source quotations, "
    "translation targets, technical terms, code, mathematical notation (such as ×, ÷, −, ≈, √), "
    "schema keys, or fixed labels for using another language. Do not require explanatory prose when "
    "the requested artifact is sufficient."
)

LANGUAGE_ROUTING_POLICY = (
    "Select by the requested task and domain regardless of the user's response language. "
    "An adapter's training language is not a required language for the user's question or the "
    "answer. Language-specific source material or dialogue specialization can guide selection, "
    "but explanations must follow the user's response-language instructions. Preserve genuine "
    "translation directions, jurisdiction limits, and domain restrictions; multilingual output "
    "does not create expertise in another domain or jurisdiction."
)
