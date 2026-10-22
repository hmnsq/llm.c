# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
استورد sys
استورد argparse

تعریف read_numbers_from_file(file_لار, ستون_پیل, ستون_پای_ټکی):
    حاول:
        numbers = []
        سره open(file_لار, 'r') كـ file:
            کرښې = file.readlines()
            پیل_شاخص = هیڅ
            لكل i, کرښه في enumerate(کرښې):
                إذا "step    1/10" في کرښه:
                    پیل_شاخص = i
                    اكسر

            إذا پیل_شاخص دی هیڅ:
                چاپ("Error: Could not find the string 'step    1/10' in the file.")
                أرجع هیڅ

            # Read 10 rows starting from the identified start row
            لكل کرښه في کرښې[پیل_شاخص:پیل_شاخص + 10]:
                # Extracting the specified columns
                شمېره = عائم(کرښه[ستون_پیل:ستون_پای_ټکی].strip())
                numbers.append(شمېره)
        أرجع numbers
    استثناء Exception كـ e:
        چاپ(f"Error reading the file: {e}")
        أرجع هیڅ

تعریف compare_numbers(read_ارزښتونه, fixed_ارزښتونه, percent_accuracy):
    لكل i في لړ(اوږدوالی(read_ارزښتونه)):
        read_ارزښت = read_ارزښتونه[i]
        fixed_ارزښت = fixed_ارزښتونه[i]
        percent_difference = ((read_ارزښت - fixed_ارزښت) / fixed_ارزښت) * 100
        چاپ(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        إذا abs(percent_difference) > percent_accuracy:
            چاپ(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            أرجع 1
    چاپ("Success: All values are within the allowed accuracy.")
    أرجع 0

تعریف اصلي():
    parser = argparse.دلیل_Parser(description='Compare numbers in a file with fixed values.')
    parser.جمع_کړه_دلیل('-f', '--file', required=ریښتیا, help='Path to the input file')
    parser.جمع_کړه_دلیل('-s', '--col_start', type=صحيح, required=ریښتیا, help='Starting column index (0-based)')
    parser.جمع_کړه_دلیل('-e', '--col_end', type=صحيح, required=ریښتیا, help='Ending column index (0-based)')
    parser.جمع_کړه_دلیل('-a', '--percent_accuracy', type=عائم, required=ریښتیا, help='Allowed percent accuracy for comparison')

    دلیلونه = parser.parse_دلیلونه()

    # Read numbers from file
    read_ارزښتونه = read_numbers_from_file(دلیلونه.file, دلیلونه.ستون_پیل, دلیلونه.ستون_پای_ټکی)
    إذا read_ارزښتونه دی هیڅ:
        أرجع 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_ارزښتونه = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    پایله = compare_numbers(read_ارزښتونه, fixed_ارزښتونه, دلیلونه.percent_accuracy)
    أرجع پایله

إذا __name__ == "__main__":
    sys.exit(اصلي())
