# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
استورد sys
استورد argparse

تعریف read_numbers_from_file(file_مسیر, ستون_شروع, ستون_انتها):
    حاول:
        numbers = []
        با open(file_مسیر, 'r') كـ file:
            سطرها = file.readlines()
            شروع_شاخص = هیچ
            لكل i, سطر في enumerate(سطرها):
                إذا "step    1/10" في سطر:
                    شروع_شاخص = i
                    اكسر

            إذا شروع_شاخص است هیچ:
                چاپ("Error: Could not find the string 'step    1/10' in the file.")
                أرجع هیچ

            # Read 10 rows starting from the identified start row
            لكل سطر في سطرها[شروع_شاخص:شروع_شاخص + 10]:
                # Extracting the specified columns
                عدد = عائم(سطر[ستون_شروع:ستون_انتها].strip())
                numbers.append(عدد)
        أرجع numbers
    استثناء Exception كـ e:
        چاپ(f"Error reading the file: {e}")
        أرجع هیچ

تعریف compare_numbers(read_مقادیر, fixed_مقادیر, percent_accuracy):
    لكل i في دامنه(طول(read_مقادیر)):
        read_مقدار = read_مقادیر[i]
        fixed_مقدار = fixed_مقادیر[i]
        percent_difference = ((read_مقدار - fixed_مقدار) / fixed_مقدار) * 100
        چاپ(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        إذا abs(percent_difference) > percent_accuracy:
            چاپ(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            أرجع 1
    چاپ("Success: All values are within the allowed accuracy.")
    أرجع 0

تعریف اصلی():
    parser = argparse.آرگومان_Parser(description='Compare numbers in a file with fixed values.')
    parser.جمع_کن_آرگومان('-f', '--file', required=درست, help='Path to the input file')
    parser.جمع_کن_آرگومان('-s', '--col_start', type=صحيح, required=درست, help='Starting column index (0-based)')
    parser.جمع_کن_آرگومان('-e', '--col_end', type=صحيح, required=درست, help='Ending column index (0-based)')
    parser.جمع_کن_آرگومان('-a', '--percent_accuracy', type=عائم, required=درست, help='Allowed percent accuracy for comparison')

    آرگومانها = parser.parse_آرگومانها()

    # Read numbers from file
    read_مقادیر = read_numbers_from_file(آرگومانها.file, آرگومانها.ستون_شروع, آرگومانها.ستون_انتها)
    إذا read_مقادیر است هیچ:
        أرجع 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_مقادیر = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    نتیجه = compare_numbers(read_مقادیر, fixed_مقادیر, آرگومانها.percent_accuracy)
    أرجع نتیجه

إذا __name__ == "__main__":
    sys.exit(اصلی())
