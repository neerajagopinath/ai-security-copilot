"""
Security Analysis Pipeline
----------------------------
Provides deterministic rule-based security analysis of source code,
combined with ML model predictions for vulnerability detection.

IMPORTANT DESIGN PRINCIPLES:
  - Rule-based findings are NEVER attributed to deep-learning predictions.
  - Model findings and rule-based findings are clearly separated.
  - No exact CWE numbers are claimed unless evidence is very strong.
  - All outputs use hedged language: "potential", "possible", "requires review".
  - This module does NOT execute, compile, or eval submitted code.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security pattern definitions
# ---------------------------------------------------------------------------

# Each pattern: (regex, short_name, description, hedged_category)
SECURITY_PATTERNS = [
    # Unsafe memory operations
    (
        re.compile(r'\b(memcpy|memmove|memset|malloc|calloc|realloc|free)\s*\(', re.IGNORECASE),
        "unsafe_memory_op",
        "Unsafe memory operation detected (memcpy/malloc/free family)",
        "Potential Memory Safety Issue",
        "High", "CWE-119", "A08:2021-Software and Data Integrity Failures",
        "Always validate malloc/calloc return values for NULL and ensure bounds checking before memory copy."
    ),
    # Buffer copy without bounds
    (
        re.compile(r'\b(strcpy|strcat|sprintf|gets|scanf)\s*\(', re.IGNORECASE),
        "unsafe_string_op",
        "Unsafe string function without bounds checking (strcpy/strcat/sprintf/gets)",
        "Possible Buffer Overflow Risk",
        "High", "CWE-120", "A03:2021-Injection",
        "Replace unsafe string functions with bounds-checked alternatives: strncpy, strncat, snprintf, fgets."
    ),
    # Safer alternatives that are still worth noting
    (
        re.compile(r'\b(strncpy|strncat|snprintf|fgets)\s*\(', re.IGNORECASE),
        "bounded_string_op",
        "Bounded string function used (strncpy/snprintf) — verify size parameter",
        "Possible Off-by-One Risk",
        "Low", "CWE-193", "A03:2021-Injection",
        "Ensure the size parameter correctly accounts for the null terminator to prevent off-by-one errors."
    ),
    # Command execution
    (
        re.compile(r'\b(system|popen|exec|execve|execl|execlp|execvp|ShellExecute|CreateProcess)\s*\(', re.IGNORECASE),
        "command_execution",
        "Command execution function detected (system/exec family)",
        "Possible Command Injection Risk",
        "High", "CWE-78", "A03:2021-Injection",
        "Avoid passing user-controlled input to system()/exec() without strict allowlist validation."
    ),
    # SQL string construction
    (
        re.compile(r'(SELECT|INSERT|UPDATE|DELETE|DROP)\s+.*?\+|sprintf.*?(SELECT|INSERT|UPDATE|DELETE)', re.IGNORECASE | re.DOTALL),
        "sql_injection",
        "Potential SQL query string construction detected",
        "Possible SQL Injection Risk",
        "High", "CWE-89", "A03:2021-Injection",
        "Use parameterized queries or prepared statements instead of building SQL strings through concatenation."
    ),
    # Hard-coded credentials
    (
        re.compile(r'(password|passwd|secret|api_key|apikey|token|credential)\s*=\s*["\'][^"\']{4,}["\']', re.IGNORECASE),
        "hardcoded_credential",
        "Possible hard-coded credential or secret detected",
        "Possible Hard-coded Secret",
        "High", "CWE-798", "A07:2021-Identification and Authentication Failures",
        "Remove hard-coded credentials. Use environment variables or secure secrets managers."
    ),
    # Weak cryptography
    (
        re.compile(r'\b(MD5|SHA1|DES|RC4|ECB)\b', re.IGNORECASE),
        "weak_crypto",
        "Weak or deprecated cryptographic algorithm reference",
        "Possible Weak Cryptography",
        "Medium", "CWE-327", "A02:2021-Cryptographic Failures",
        "Replace weak cryptographic algorithms (MD5, SHA-1, DES) with modern standards (SHA-256, AES-GCM)."
    ),
    # Integer overflow indicators
    (
        re.compile(r'\b(INT_MAX|UINT_MAX|SIZE_MAX)\b'),
        "integer_boundary",
        "Integer boundary constant usage — verify overflow handling",
        "Possible Integer Overflow/Underflow Risk",
        "Low", "CWE-190", "A03:2021-Injection",
        "Ensure arithmetic operations on integers do not result in overflow or underflow."
    ),
    # Null pointer dereference risk
    (
        re.compile(r'=\s*malloc\s*\(.*?\)\s*;(?!.*?if\s*\()'),
        "null_ptr_risk",
        "malloc() return value may not be checked for NULL",
        "Possible Null Pointer Dereference",
        "Medium", "CWE-476", "A08:2021-Software and Data Integrity Failures",
        "Always check the return value of malloc()/calloc() for NULL before use."
    ),
    # Format string
    (
        re.compile(r'\b(printf|fprintf|sprintf|syslog)\s*\([^"]*[^,)]+\)', re.IGNORECASE),
        "format_string",
        "Possible format string vulnerability — user input as format argument",
        "Possible Format String Vulnerability",
        "High", "CWE-134", "A03:2021-Injection",
        "Never pass user-controlled strings directly as format arguments. Use printf(\"%s\", input)."
    ),
    # Dangerous file operations
    (
        re.compile(r'\b(fopen|open|creat)\s*\(', re.IGNORECASE),
        "file_operation",
        "File operation detected — verify path validation and permissions",
        "Possible Insecure File Operation",
        "Medium", "CWE-22", "A01:2021-Broken Access Control",
        "Validate and canonicalize file paths before use. Restrict file access."
    ),
    # Missing input validation (common patterns)
    (
        re.compile(r'\b(atoi|atol|atof)\s*\(', re.IGNORECASE),
        "unvalidated_input",
        "String-to-integer conversion without validation (atoi/atol/atof)",
        "Possible Missing Input Validation",
        "Low", "CWE-20", "A03:2021-Injection",
        "Validate input before conversion. Consider using strtol() with error checking."
    ),
    # Use after free pattern hint
    (
        re.compile(r'free\s*\([^)]+\)\s*;[^;]{0,80}?(?:->|\[|\.)'),
        "use_after_free_hint",
        "Possible use-after-free pattern (pointer dereferenced after free())",
        "Possible Use-After-Free Risk",
        "High", "CWE-416", "A08:2021-Software and Data Integrity Failures",
        "Set pointers to NULL immediately after freeing them to prevent use-after-free."
    ),
]


@dataclass
class RuleMatch:
    pattern_name: str
    description: str
    hedged_category: str
    matched_text: str
    severity: str
    line_number: int
    cwe: str
    owasp: str
    recommendation: str


@dataclass
class AnalysisResult:
    """Complete analysis result for a code snippet."""
    # Input metadata
    language: str
    model_used: str

    # ML model output
    prediction: str           # "Potentially Vulnerable" or "No Vulnerability Detected"
    vulnerability_probability: float
    confidence: float
    model_mode: str           # "trained", "demo", "fallback"

    # Rule-based findings
    rule_matches: List[RuleMatch] = field(default_factory=list)
    suspicious_patterns: List[str] = field(default_factory=list)

    # Combined analysis
    potential_category: str = "Unknown"
    explanation: str = ""
    recommendations: List[str] = field(default_factory=list)
    structured_findings: List[Dict[str, Any]] = field(default_factory=list)
    suggested_code: str = ""
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# Rule-based analysis
# ---------------------------------------------------------------------------

def run_rule_analysis(code: str) -> List[RuleMatch]:
    """
    Run all security patterns against a code string.
    Returns a list of RuleMatch objects for each detected pattern.
    """
    matches = []
    for pattern, name, description, category, severity, cwe, owasp, rec in SECURITY_PATTERNS:
        for found in pattern.finditer(code):
            matched_text = found.group(0)[:100]  # cap at 100 chars
            line_number = code[:found.start()].count('\n') + 1
            matches.append(
                RuleMatch(
                    pattern_name=name,
                    description=description,
                    hedged_category=category,
                    matched_text=matched_text,
                    severity=severity,
                    line_number=line_number,
                    cwe=cwe,
                    owasp=owasp,
                    recommendation=rec,
                )
            )
    return matches


def derive_category_from_rules(rule_matches: List[RuleMatch]) -> str:
    """
    Derive the most prominent potential vulnerability category
    from rule matches. Uses priority ordering.
    """
    if not rule_matches:
        return "No Pattern Detected"

    # Priority order of categories
    priority = [
        "command_execution",
        "sql_injection",
        "hardcoded_credential",
        "unsafe_string_op",
        "unsafe_memory_op",
        "null_ptr_risk",
        "use_after_free_hint",
        "format_string",
        "weak_crypto",
        "unvalidated_input",
        "file_operation",
        "bounded_string_op",
        "integer_boundary",
    ]

    match_names = {m.pattern_name for m in rule_matches}
    for name in priority:
        if name in match_names:
            for m in rule_matches:
                if m.pattern_name == name:
                    return m.hedged_category

    return rule_matches[0].hedged_category


def build_recommendations(rule_matches: List[RuleMatch], prediction: str) -> List[str]:
    """Build actionable security recommendations based on findings."""
    recs = []

    names = {m.pattern_name for m in rule_matches}

    if "unsafe_string_op" in names:
        recs.append(
            "Replace unsafe string functions (strcpy, strcat, sprintf, gets) with "
            "bounds-checked alternatives: strncpy, strncat, snprintf, fgets."
        )
    if "unsafe_memory_op" in names:
        recs.append(
            "Always validate malloc/calloc return values for NULL. "
            "Ensure every allocated buffer is freed exactly once."
        )
    if "command_execution" in names:
        recs.append(
            "Avoid passing user-controlled input to system()/popen()/exec() without "
            "strict allowlist validation. Prefer safer library alternatives."
        )
    if "sql_injection" in names:
        recs.append(
            "Use parameterized queries or prepared statements instead of building "
            "SQL strings through concatenation."
        )
    if "hardcoded_credential" in names:
        recs.append(
            "Remove hard-coded credentials. Use environment variables, secrets "
            "managers, or configuration files excluded from version control."
        )
    if "weak_crypto" in names:
        recs.append(
            "Replace weak cryptographic algorithms (MD5, SHA-1, DES, RC4) with "
            "modern standards: SHA-256, AES-GCM, or ChaCha20-Poly1305."
        )
    if "null_ptr_risk" in names:
        recs.append(
            "Always check the return value of malloc()/calloc() for NULL before use."
        )
    if "format_string" in names:
        recs.append(
            "Never pass user-controlled strings directly as format arguments. "
            "Use printf(\"%s\", user_input) instead of printf(user_input)."
        )
    if "unvalidated_input" in names:
        recs.append(
            "Validate and sanitize all external input before conversion. "
            "Consider strtol() with error checking instead of atoi()."
        )
    if "file_operation" in names:
        recs.append(
            "Validate and canonicalize file paths before use. "
            "Restrict file access using least-privilege principles."
        )

    if prediction == "Potentially Vulnerable" and not recs:
        recs.append(
            "The model flagged this code as potentially vulnerable. "
            "Conduct a thorough manual security review and use a static analysis tool."
        )

    if not recs:
        recs.append(
            "No specific rule-based issues detected. Continue with standard "
            "security review practices and static analysis tools."
        )

    recs.append(
        "Run static analysis tools (e.g., Coverity, CodeQL, Semgrep) for deeper inspection."
    )

    return recs


def generate_suggested_code(code: str, rule_matches: List[RuleMatch]) -> str:
    """
    Apply simple, safe code transformations to suggest more secure alternatives.
    These are pattern-based textual replacements — not guaranteed correct.
    Always requires developer review.
    """
    if not rule_matches:
        return ""

    names = {m.pattern_name for m in rule_matches}
    suggested = code

    # Replace gets → fgets (simple substitution hint)
    if "unsafe_string_op" in names:
        suggested = re.sub(r'\bgets\s*\(', 'fgets(', suggested)
        suggested = re.sub(r'\bstrcpy\s*\(', '/* TODO: replace with strncpy */ strcpy(', suggested)
        suggested = re.sub(r'\bstrcat\s*\(', '/* TODO: replace with strncat */ strcat(', suggested)
        suggested = re.sub(r'\bsprintf\s*\(', 'snprintf(', suggested)

    if "weak_crypto" in names:
        suggested = re.sub(r'\bMD5\b', '/* TODO: replace MD5 with SHA-256 */ MD5', suggested)
        suggested = re.sub(r'\bSHA1\b', '/* TODO: replace SHA-1 with SHA-256 */ SHA1', suggested)

    if suggested == code:
        return ""  # No changes made

    return (
        "/* AI Security Copilot - Suggested secure rewrite (REQUIRES DEVELOPER REVIEW):\n"
        "   These are automated hints, not production-ready fixes. */\n\n"
        + suggested
    )


# ---------------------------------------------------------------------------
# Main analysis service
# ---------------------------------------------------------------------------

class SecurityAnalysisService:
    """
    Main analysis service that combines ML model predictions
    with deterministic rule-based security analysis.
    """

    DISCLAIMER = (
        "DISCLAIMER: This analysis is provided for educational and informational "
        "purposes only. It combines machine-learning model output with deterministic "
        "security-pattern detection. Rule-based findings are not deep-learning "
        "predictions. The absence of findings does NOT guarantee security. Always "
        "conduct a thorough manual security review and use professional static analysis "
        "tools before deploying code in production. The suggested code improvements "
        "are automated hints and require developer review."
    )

    def __init__(self, model_manager: Optional[Any] = None):
        """
        Args:
            model_manager: Optional ModelManager instance providing ML predictions.
                           If None, rule-based analysis only is performed.
        """
        self.model_manager = model_manager

    def analyze(
        self,
        code: str,
        language: str = "C/C++",
        model_name: str = "bilstm",
    ) -> Dict[str, Any]:
        """
        Full vulnerability analysis pipeline.

        Args:
            code: Source code string to analyze.
            language: Programming language (for display).
            model_name: ML model to use ("bilstm" or "graphcodebert").

        Returns:
            Dictionary with all analysis fields.
        """
        if not isinstance(code, str) or not code.strip():
            return self._empty_result(language, model_name)

        # 1. Run deterministic rule-based analysis
        rule_matches = run_rule_analysis(code)
        suspicious_patterns = [m.description for m in rule_matches]

        # 2. Get ML model prediction
        ml_prob = 0.0
        model_mode = "fallback"

        if self.model_manager is not None:
            try:
                result = self.model_manager.predict(code, model_name)
                ml_prob = result.get("probability", 0.0)
                model_mode = result.get("mode", "trained")
            except Exception as exc:
                logger.warning("Model prediction failed, using rule-based only: %s", exc)
                ml_prob = self._rule_based_prob(rule_matches)
                model_mode = "fallback"
        else:
            ml_prob = self._rule_based_prob(rule_matches)
            model_mode = "fallback"

        # 3. Combine: final probability
        final_prob = ml_prob
        if model_mode == "fallback" and rule_matches:
            # Boost probability based on number and severity of rule findings
            rule_boost = min(0.3, len(rule_matches) * 0.08)
            final_prob = min(0.95, final_prob + rule_boost)

        # 4. Prediction label
        threshold = 0.5
        prediction = "Potentially Vulnerable" if final_prob >= threshold else "No Vulnerability Detected"
        confidence = final_prob if final_prob >= threshold else (1.0 - final_prob)

        # 5. Derive category
        if rule_matches:
            potential_category = derive_category_from_rules(rule_matches)
        elif prediction == "Potentially Vulnerable":
            potential_category = "General Vulnerability Pattern (Model-detected)"
        else:
            potential_category = "No Specific Pattern Identified"

        # 6. Explanation
        explanation = self._build_explanation(
            prediction, ml_prob, rule_matches, model_mode
        )

        # 7. Recommendations
        recommendations = build_recommendations(rule_matches, prediction)

        # 8. Structured Findings
        structured_findings = []
        for m in rule_matches:
            structured_findings.append({
                "type": m.hedged_category,
                "severity": m.severity,
                "line": m.line_number,
                "cwe": m.cwe,
                "owasp": m.owasp,
                "reason": m.description,
                "recommendation": m.recommendation
            })

        # 9. Suggested code
        suggested_code = generate_suggested_code(code, rule_matches)

        return {
            "model": model_name,
            "language": language,
            "prediction": prediction,
            "vulnerability_probability": round(final_prob, 4),
            "confidence": round(confidence, 4),
            "model_mode": model_mode,
            "potential_category": potential_category,
            "suspicious_patterns": suspicious_patterns,
            "rule_matches_count": len(rule_matches),
            "explanation": explanation,
            "recommendations": recommendations,
            "structured_findings": structured_findings,
            "suggested_code": suggested_code,
            "disclaimer": self.DISCLAIMER,
        }

    def _rule_based_prob(self, rule_matches: List[RuleMatch]) -> float:
        """Heuristic probability based solely on rule matches."""
        if not rule_matches:
            return 0.15  # Low baseline
        severity_map = {
            "command_execution": 0.85,
            "sql_injection": 0.80,
            "hardcoded_credential": 0.75,
            "unsafe_string_op": 0.70,
            "null_ptr_risk": 0.65,
            "use_after_free_hint": 0.70,
            "format_string": 0.65,
            "unsafe_memory_op": 0.60,
            "weak_crypto": 0.55,
            "unvalidated_input": 0.50,
            "file_operation": 0.40,
            "bounded_string_op": 0.30,
            "integer_boundary": 0.35,
        }
        max_prob = max(
            severity_map.get(m.pattern_name, 0.40) for m in rule_matches
        )
        return max_prob

    def _build_explanation(
        self,
        prediction: str,
        ml_prob: float,
        rule_matches: List[RuleMatch],
        model_mode: str,
    ) -> str:
        parts = []

        if model_mode == "trained":
            parts.append(
                f"The trained Bi-LSTM model assigned a vulnerability probability of "
                f"{ml_prob:.0%} to this code snippet."
            )
        elif model_mode == "demo":
            parts.append(
                f"A demo-trained Bi-LSTM model (NOT a fully trained model) assigned "
                f"a probability of {ml_prob:.0%}. Treat this as approximate only."
            )
        else:
            parts.append(
                "No trained model checkpoint is available. The vulnerability "
                "probability was estimated from deterministic security-rule analysis only."
            )

        if rule_matches:
            rule_names = ", ".join(m.hedged_category for m in rule_matches[:3])
            parts.append(
                f"Rule-based analysis detected {len(rule_matches)} potential "
                f"security pattern(s): {rule_names}."
            )
        else:
            parts.append(
                "Rule-based analysis did not detect any known high-risk security patterns."
            )

        if prediction == "Potentially Vulnerable":
            parts.append(
                "This code requires a careful security review before use in production."
            )
        else:
            parts.append(
                "No strong vulnerability signals were detected, but this does not "
                "guarantee the code is secure."
            )

        return " ".join(parts)

    def _empty_result(self, language: str, model_name: str) -> Dict[str, Any]:
        return {
            "model": model_name,
            "language": language,
            "prediction": "No Vulnerability Detected",
            "vulnerability_probability": 0.0,
            "confidence": 1.0,
            "model_mode": "fallback",
            "potential_category": "Empty Input",
            "suspicious_patterns": [],
            "rule_matches_count": 0,
            "explanation": "No source code was provided for analysis.",
            "recommendations": ["Please provide valid source code for analysis."],
            "structured_findings": [],
            "suggested_code": "",
            "disclaimer": self.DISCLAIMER,
        }
