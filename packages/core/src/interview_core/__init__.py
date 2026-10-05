"""Core algorithms for the AI Interview Coach.

Module-to-patent-element map (abstract of application 202541122226):

- E2 facial recognition ........ :mod:`interview_core.face`
- E3 voice authentication ...... :mod:`interview_core.voice`
- E4 lip-sync verification ..... :mod:`interview_core.lipsync`
- E5 eye tracking .............. :mod:`interview_core.gaze`
- E6 NLP (resume, questions, evaluation) :mod:`interview_core.nlp`
- E7 security monitoring ....... :mod:`interview_core.security`
- E8 technical assessment ...... :mod:`interview_core.codeexec`
- E9 performance evaluation .... :mod:`interview_core.report`, :mod:`interview_core.delivery`

:mod:`interview_core.legacy` holds behaviour-identical re-implementations of the
prototype algorithms (``legacy/desktop/main.py``), kept as the regression reference.
"""

__version__ = "0.1.0"
