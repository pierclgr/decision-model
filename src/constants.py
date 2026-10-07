MAX_LETTER_OPTIONS: int = 26

# final line of the prompt: System One reads the letter logits right after it
SYSTEM_ONE_INSTRUCTION: str = "Answer with the letter of the correct option only."
# System Two generates its answer in a fixed block, parsed by `SystemTwoModel`
SYSTEM_TWO_INSTRUCTION: str = (
    "Answer exactly in this format:\n<answer>\n[letter of the correct option]"
    "\n</answer>\nFor example, if the correct option is A:\n<answer>\nA\n</answer>"
)
