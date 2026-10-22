# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
استورد sys
استورد argparse

تعریف read_numbers_from_file(file_رستہ, خانہ_شروع, خانہ_سرا):
    حاول:
        numbers = []
        نال open(file_رستہ, 'r') كـ file:
            سطریں = file.readlines()
            شروع_اشاریہ = کجھ_نئیں
            لكل i, سطر في enumerate(سطریں):
                إذا "step    1/10" في سطر:
                    شروع_اشاریہ = i
                    اكسر

            إذا شروع_اشاریہ اے کجھ_نئیں:
                چھاپو("Error: Could not find the string 'step    1/10' in the file.")
                أرجع کجھ_نئیں

            # Read 10 rows starting from the identified start row
            لكل سطر في سطریں[شروع_اشاریہ:شروع_اشاریہ + 10]:
                # Extracting the specified columns
                نمبر = عائم(سطر[خانہ_شروع:خانہ_سرا].strip())
                numbers.append(نمبر)
        أرجع numbers
    استثناء Exception كـ e:
        چھاپو(f"Error reading the file: {e}")
        أرجع کجھ_نئیں

تعریف compare_numbers(read_اقدار, fixed_اقدار, percent_accuracy):
    لكل i في سلسلہ(لمبائی(read_اقدار)):
        read_قدر = read_اقدار[i]
        fixed_قدر = fixed_اقدار[i]
        percent_difference = ((read_قدر - fixed_قدر) / fixed_قدر) * 100
        چھاپو(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        إذا abs(percent_difference) > percent_accuracy:
            چھاپو(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            أرجع 1
    چھاپو("Success: All values are within the allowed accuracy.")
    أرجع 0

تعریف مکھ():
    parser = argparse.دلیل_Parser(description='Compare numbers in a file with fixed values.')
    parser.جوڑو_دلیل('-f', '--file', required=سچ, help='Path to the input file')
    parser.جوڑو_دلیل('-s', '--col_start', type=صحيح, required=سچ, help='Starting column index (0-based)')
    parser.جوڑو_دلیل('-e', '--col_end', type=صحيح, required=سچ, help='Ending column index (0-based)')
    parser.جوڑو_دلیل('-a', '--percent_accuracy', type=عائم, required=سچ, help='Allowed percent accuracy for comparison')

    دلیلاں = parser.parse_دلیلاں()

    # Read numbers from file
    read_اقدار = read_numbers_from_file(دلیلاں.file, دلیلاں.خانہ_شروع, دلیلاں.خانہ_سرا)
    إذا read_اقدار اے کجھ_نئیں:
        أرجع 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_اقدار = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    نتیجہ = compare_numbers(read_اقدار, fixed_اقدار, دلیلاں.percent_accuracy)
    أرجع نتیجہ

إذا __name__ == "__main__":
    sys.exit(مکھ())
