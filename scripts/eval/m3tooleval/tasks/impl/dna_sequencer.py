import itertools
from ..base import Task, register_task_iterator, ToolType

# Cryptobotanist's Plant DNA Sequencer
# Tool 1: DNA Nucleotide Counter
def count_nucleotides(dna_sequence):
    from collections import Counter
    return dict(Counter(dna_sequence))

# Tool 2: DNA to mRNA Transcriber
def transcribe_dna_to_mrna(dna_sequence):
    transcription_map = str.maketrans('ACGT', 'UGCA')
    return dna_sequence.translate(transcription_map)

# Tool 3: mRNA to Amino Acid Translator
def translate_mrna_to_amino_acid(mrna_sequence):
    # Simplified codon to amino acid mapping
    codon_map = {'AUG': 'Methionine', 'UUU': 'Phenylalanine', 'UUC': 'Phenylalanine'}
    amino_acids = [codon_map.get(mrna_sequence[i:i+3], 'X') for i in range(0, len(mrna_sequence), 3)]
    return '-'.join(amino_acids)

# Tool 4: Find Max Occurring Nucleotide
def find_max_nucleotide(*args):
    # args are assumed to be in the form of (k1, v1, k2, v2, ..., kn, vn)
    nucleotide_counts = dict(zip(args[::2], args[1::2]))
    max_nucleotide = max(nucleotide_counts, key=nucleotide_counts.get)
    return (max_nucleotide, nucleotide_counts[max_nucleotide])

# Tool 5: Check if Sequence is Valid (Only A, C, G, T are valid)
def is_valid_dna_sequence(dna_sequence):
    return set(dna_sequence).issubset(set('ACGT'))

# Tool 6: Reverse Transcribe mRNA to DNA
def reverse_transcribe_mrna_to_dna(mrna_sequence):
    reverse_transcription_map = str.maketrans('UCAG', 'AGTC')
    return mrna_sequence.translate(reverse_transcription_map)

# Register additional tools

# Registering the tools and task for Task 5
CUR_TASK_NAME = 'cryptobotanists_plant_dna_sequencer'
CUR_TOOLS = {
    'count_nucleotides': ToolType(
        name='count_nucleotides',
        description='Counts the occurrences of each nucleotide in a DNA sequence. Arguments: dna_sequence (str)',
        function=count_nucleotides,
        fn_signature='count_nucleotides(dna_sequence: str) -> dict'
    ),
    'transcribe_dna_to_mrna': ToolType(
        name='transcribe_dna_to_mrna',
        description='Transcribes DNA sequence to mRNA. Arguments: dna_sequence (str)',
        function=transcribe_dna_to_mrna,
        fn_signature='transcribe_dna_to_mrna(dna_sequence: str) -> str'
    ),
    'translate_mrna_to_amino_acid': ToolType(
        name='translate_mrna_to_amino_acid',
        description='Translates mRNA sequence to a chain of amino acids. Arguments: mrna_sequence (str)',
        function=translate_mrna_to_amino_acid,
        fn_signature='translate_mrna_to_amino_acid(mrna_sequence: str) -> str'
    ),
    'find_max_nucleotide': ToolType(
        name='find_max_nucleotide',
        description='Return the nucleotide (str) with the maximum count (int). Arguments: nucleotide_counts in the form of (k1, v1, k2, v2, ..., kn, vn)',
        function=find_max_nucleotide,
        fn_signature='find_max_nucleotide(*args) -> (str, int)'
    ),
    'is_valid_dna_sequence': ToolType(
        name='is_valid_dna_sequence',
        description='Checks if the DNA sequence is valid. Arguments: dna_sequence (str)',
        function=is_valid_dna_sequence,
        fn_signature='is_valid_dna_sequence(dna_sequence: str) -> bool'
    ),
    'reverse_transcribe_mrna_to_dna': ToolType(
        name='reverse_transcribe_mrna_to_dna',
        description='Reverse transcribes mRNA sequence to DNA. Arguments: mrna_sequence (str)',
        function=reverse_transcribe_mrna_to_dna,
        fn_signature='reverse_transcribe_mrna_to_dna(mrna_sequence: str) -> str'
    )
}

TASKS = []

# Task 1: Find the most common nucleotide in a DNA sequence and transcribe that DNA to mRNA.
def task1_complex_operation(dna_sequence):
    counts = count_nucleotides(dna_sequence)
    max_nucleotide, _ = find_max_nucleotide(*list(
        sum(map(list, counts.items()), [])
    ))
    return transcribe_dna_to_mrna(max_nucleotide * 3)

TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/find_most_common_nucleotide_and_transcribe",
        tools=CUR_TOOLS,
        instruction="Find the most common nucleotide in the DNA sequence 'AGCTAGCCGATGCA' and transcribe that nucleotide (repeated three times) to mRNA.",
        expected_output=task1_complex_operation('AGCTAGCCGATGCA'),
        is_single_tool_task=False
    )
)

# Task 2: Check if a DNA sequence is valid, and if so, transcribe to mRNA and translate to an amino acid sequence.
def task2_complex_operation(dna_sequence):
    if is_valid_dna_sequence(dna_sequence):
        mrna = transcribe_dna_to_mrna(dna_sequence)
        return translate_mrna_to_amino_acid(mrna)
    return "Invalid DNA sequence"

TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/validate_transcribe_translate",
        tools=CUR_TOOLS,
        instruction="Check if the DNA sequence 'AGCTTX' is valid, and if so, transcribe it to mRNA and translate that to an amino acid sequence. Otherwise, answer 'Invalid DNA sequence'.",
        expected_output=task2_complex_operation('AGCTTX'),
        is_single_tool_task=False
    )
)

# Task 3: For a given mRNA sequence, reverse transcribe to DNA, count the nucleotides, and find the maximum occurring nucleotide.
def task3_complex_operation(mrna_sequence):
    dna_sequence = reverse_transcribe_mrna_to_dna(mrna_sequence)
    counts = count_nucleotides(dna_sequence)
    max_nucleotide, max_count = find_max_nucleotide(*list(
        sum(map(list, counts.items()), [])
    ))
    return max_nucleotide

TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/reverse_transcribe_count_max_nucleotide",
        tools=CUR_TOOLS,
        instruction="For the mRNA sequence 'AUGCUUUUC', reverse transcribe it to DNA, count the nucleotides, and find the maximum occurring nucleotide. Answer the maximum occurring nucleotide, not the count.",
        expected_output=task3_complex_operation('AUGCUUUUC'),
        is_single_tool_task=False
    )
)

# Task 4: Transcribe, Translate, and Determine Amino Acid Length
def task4_complex_operation(dna_sequence):
    mrna = transcribe_dna_to_mrna(dna_sequence)
    amino_acid_sequence = translate_mrna_to_amino_acid(mrna)
    return len(amino_acid_sequence.split('-'))

TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/transcribe_translate_amino_acid_length",
        tools=CUR_TOOLS,
        instruction="Transcribe the DNA sequence 'AGCTAGCGTA' to mRNA, translate it to an amino acid sequence, and determine the length of the amino acid sequence.",
        expected_output=task4_complex_operation('AGCTAGCGTA'),
        is_single_tool_task=False
    )
)

# Task 5: Reverse Transcription and Nucleotide Counting for Palindromic Sequence
def task5_complex_operation(mrna_sequence):
    dna_sequence = reverse_transcribe_mrna_to_dna(mrna_sequence)
    return count_nucleotides(dna_sequence)

TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/reverse_transcribe_palindrome_check",
        tools=CUR_TOOLS,
        instruction="Reverse transcribe the mRNA sequence 'AUGCGU' to DNA, and count the nucleotides in the DNA sequence.",
        expected_output=task5_complex_operation('AUGCGU'),
        is_single_tool_task=False
    )
)

# Task 6: Validating Multiple DNA Sequences and Finding the Longest Valid Sequence
def task6_complex_operation(dna_sequences):
    valid_sequences = [seq for seq in dna_sequences if is_valid_dna_sequence(seq)]
    return max(valid_sequences, key=len) if valid_sequences else "No valid DNA sequences"

TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/validate_find_longest_valid_sequence",
        tools=CUR_TOOLS,
        instruction="Given the DNA sequences ['AGCTAG', 'XYZABC', 'GTCAGT'], check which are valid and find the longest valid DNA sequence.",
        expected_output=task6_complex_operation(['AGCTAG', 'XYZABC', 'GTCAGT']),
        is_single_tool_task=False
    )
)

# Task 7: Transcription, Translation, and Identifying a Specific Amino Acid
def task7_complex_operation(dna_sequence, target_amino_acid):
    mrna = transcribe_dna_to_mrna(dna_sequence)
    amino_acid_sequence = translate_mrna_to_amino_acid(mrna)
    return target_amino_acid in amino_acid_sequence

TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/transcribe_translate_find_amino_acid",
        tools=CUR_TOOLS,
        instruction="Transcribe the DNA sequence 'AGCTAGCGTA' to mRNA, translate it to an amino acid sequence, and check if 'Methionine' is in the amino acid sequence. Answer 'True' or 'False'.",
        expected_output=task7_complex_operation('AGCTAGCGTA', 'Methionine'),
        is_single_tool_task=False
    )
)

# Task 8: Find the Most Common Nucleotide in a Combined DNA Sequence
def task8_complex_operation(dna_sequences):
    combined_sequence = ''.join(dna_sequences)
    counts = count_nucleotides(combined_sequence)
    return find_max_nucleotide(*list(sum(map(list, counts.items()), [])))

TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/find_most_common_nucleotide_combined_sequence",
        tools=CUR_TOOLS,
        instruction="Combine the DNA sequences ['AGCT', 'CCGA', 'TTAG'] and find the most common nucleotide in the combined sequence.",
        expected_output=task8_complex_operation(['AGCT', 'CCGA', 'TTAG']),
        is_single_tool_task=False
    )
)

register_task_iterator(TASKS, len(TASKS))
print(f"**** {len(TASKS)} tasks registered for {CUR_TASK_NAME} ****")
