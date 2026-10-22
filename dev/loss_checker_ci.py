# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
درآمد sys
درآمد argparse

تعریف read_numbers_from_file(file_راستہ, ستون_آغاز, ستون_انتہا):
    کوشش:
        numbers = []
        ساتھ open(file_راستہ, 'r') بطور file:
            سطریں = file.readlines()
            آغاز_اشاریہ = کچھ_نہیں
            برائے i, سطر میں enumerate(سطریں):
                اگر "step    1/10" میں سطر:
                    آغاز_اشاریہ = i
                    توڑو

            اگر آغاز_اشاریہ ہے کچھ_نہیں:
                چھاپو("Error: Could not find the string 'step    1/10' in the file.")
                واپس کچھ_نہیں

            # Read 10 rows starting from the identified start row
            برائے سطر میں سطریں[آغاز_اشاریہ:آغاز_اشاریہ + 10]:
                # Extracting the specified columns
                نمبر = اعشاری(سطر[ستون_آغاز:ستون_انتہا].strip())
                numbers.append(نمبر)
        واپس numbers
    استثنا Exception بطور e:
        چھاپو(f"Error reading the file: {e}")
        واپس کچھ_نہیں

تعریف compare_numbers(read_اقدار, fixed_اقدار, percent_accuracy):
    برائے i میں سلسلہ(لمبائی(read_اقدار)):
        read_قدر = read_اقدار[i]
        fixed_قدر = fixed_اقدار[i]
        percent_difference = ((read_قدر - fixed_قدر) / fixed_قدر) * 100
        چھاپو(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        اگر abs(percent_difference) > percent_accuracy:
            چھاپو(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            واپس 1
    چھاپو("Success: All values are within the allowed accuracy.")
    واپس 0

تعریف مین():
    parser = argparse.دلیل_Parser(description='Compare numbers in a file with fixed values.')
    parser.جوڑو_دلیل('-f', '--file', required=سچ, help='Path to the input file')
    parser.جوڑو_دلیل('-s', '--col_start', type=صحیح, required=سچ, help='Starting column index (0-based)')
    parser.جوڑو_دلیل('-e', '--col_end', type=صحیح, required=سچ, help='Ending column index (0-based)')
    parser.جوڑو_دلیل('-a', '--percent_accuracy', type=اعشاری, required=سچ, help='Allowed percent accuracy for comparison')

    دلائل = parser.parse_دلائل()

    # Read numbers from file
    read_اقدار = read_numbers_from_file(دلائل.file, دلائل.ستون_آغاز, دلائل.ستون_انتہا)
    اگر read_اقدار ہے کچھ_نہیں:
        واپس 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_اقدار = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    نتیجہ = compare_numbers(read_اقدار, fixed_اقدار, دلائل.percent_accuracy)
    واپس نتیجہ

اگر __name__ == "__main__":
    sys.exit(مین())
