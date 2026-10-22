# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
ಆಮದು sys
ಆಮದು argparse

ವ್ಯಾಖ್ಯಾನ read_numbers_from_file(file_ಮಾರ್ಗ, ಲಂಬಸಾಲು_ಆರಂಭ, ಲಂಬಸಾಲು_ಕೊನೆ):
    ಪ್ರಯತ್ನಿಸು:
        numbers = []
        ಜೊತೆ open(file_ಮಾರ್ಗ, 'r') ಇವ file:
            ಸಾಲುಗಳು = file.readlines()
            ಆರಂಭ_ಸೂಚ್ಯಂಕ = ಏನೂಇಲ್ಲ
            ಗಾಗಿ i, ಸಾಲು ಒಳಗೆ enumerate(ಸಾಲುಗಳು):
                ಆದರೆ "step    1/10" ಒಳಗೆ ಸಾಲು:
                    ಆರಂಭ_ಸೂಚ್ಯಂಕ = i
                    ನಿಲ್ಲಿಸು

            ಆದರೆ ಆರಂಭ_ಸೂಚ್ಯಂಕ ಆಗಿರುವ ಏನೂಇಲ್ಲ:
                ಮುದ್ರಿಸು("Error: Could not find the string 'step    1/10' in the file.")
                ಹಿಂತಿರುಗಿ ಏನೂಇಲ್ಲ

            # Read 10 rows starting from the identified start row
            ಗಾಗಿ ಸಾಲು ಒಳಗೆ ಸಾಲುಗಳು[ಆರಂಭ_ಸೂಚ್ಯಂಕ:ಆರಂಭ_ಸೂಚ್ಯಂಕ + 10]:
                # Extracting the specified columns
                ಸಂಖ್ಯೆ = ದಶಮಾಂಶ(ಸಾಲು[ಲಂಬಸಾಲು_ಆರಂಭ:ಲಂಬಸಾಲು_ಕೊನೆ].strip())
                numbers.append(ಸಂಖ್ಯೆ)
        ಹಿಂತಿರುಗಿ numbers
    ಅಪವಾದ Exception ಇವ e:
        ಮುದ್ರಿಸು(f"Error reading the file: {e}")
        ಹಿಂತಿರುಗಿ ಏನೂಇಲ್ಲ

ವ್ಯಾಖ್ಯಾನ compare_numbers(read_ಮೌಲ್ಯಗಳು, fixed_ಮೌಲ್ಯಗಳು, percent_accuracy):
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(ಉದ್ದ(read_ಮೌಲ್ಯಗಳು)):
        read_ಮೌಲ್ಯ = read_ಮೌಲ್ಯಗಳು[i]
        fixed_ಮೌಲ್ಯ = fixed_ಮೌಲ್ಯಗಳು[i]
        percent_difference = ((read_ಮೌಲ್ಯ - fixed_ಮೌಲ್ಯ) / fixed_ಮೌಲ್ಯ) * 100
        ಮುದ್ರಿಸು(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        ಆದರೆ abs(percent_difference) > percent_accuracy:
            ಮುದ್ರಿಸು(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            ಹಿಂತಿರುಗಿ 1
    ಮುದ್ರಿಸು("Success: All values are within the allowed accuracy.")
    ಹಿಂತಿರುಗಿ 0

ವ್ಯಾಖ್ಯಾನ ಮುಖ್ಯ():
    parser = argparse.ವಾದ_Parser(description='Compare numbers in a file with fixed values.')
    parser.ಕೂಡಿಸು_ವಾದ('-f', '--file', required=ನಿಜ, help='Path to the input file')
    parser.ಕೂಡಿಸು_ವಾದ('-s', '--col_start', type=ಪೂರ್ಣಾಂಕ, required=ನಿಜ, help='Starting column index (0-based)')
    parser.ಕೂಡಿಸು_ವಾದ('-e', '--col_end', type=ಪೂರ್ಣಾಂಕ, required=ನಿಜ, help='Ending column index (0-based)')
    parser.ಕೂಡಿಸು_ವಾದ('-a', '--percent_accuracy', type=ದಶಮಾಂಶ, required=ನಿಜ, help='Allowed percent accuracy for comparison')

    ವಾದಗಳು = parser.parse_ವಾದಗಳು()

    # Read numbers from file
    read_ಮೌಲ್ಯಗಳು = read_numbers_from_file(ವಾದಗಳು.file, ವಾದಗಳು.ಲಂಬಸಾಲು_ಆರಂಭ, ವಾದಗಳು.ಲಂಬಸಾಲು_ಕೊನೆ)
    ಆದರೆ read_ಮೌಲ್ಯಗಳು ಆಗಿರುವ ಏನೂಇಲ್ಲ:
        ಹಿಂತಿರುಗಿ 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_ಮೌಲ್ಯಗಳು = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    ಫಲಿತಾಂಶ = compare_numbers(read_ಮೌಲ್ಯಗಳು, fixed_ಮೌಲ್ಯಗಳು, ವಾದಗಳು.percent_accuracy)
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ

ಆದರೆ __name__ == "__main__":
    sys.exit(ಮುಖ್ಯ())
