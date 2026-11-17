class AinsightError(Exception):
    error_code = "AINSIGHT_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidSelectionError(AinsightError):
    error_code = "INVALID_SELECTION"


class CacheContextMissingError(AinsightError):
    error_code = "CACHE_CONTEXT_MISSING"


class AIOutputValidationError(AinsightError):
    error_code = "AI_OUTPUT_VALIDATION_FAILED"


class ModelProviderError(AinsightError):
    error_code = "AI_PROVIDER_ERROR"

