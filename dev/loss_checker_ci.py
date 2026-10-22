# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
استورد sys
استورد argparse

وصف read_numbers_from_file(file_رستو, کانو_شروعات, کانو_پڇاڙي):
    حاول:
        numbers = []
        سان open(file_رستو, 'r') كـ file:
            سٽون = file.readlines()
            شروعات_اشاريو = ڪجهه_نه
            لكل i, سٽ في enumerate(سٽون):
                إذا "step    1/10" في سٽ:
                    شروعات_اشاريو = i
                    اكسر

            إذا شروعات_اشاريو آهي ڪجهه_نه:
                ڇاپيو("Error: Could not find the string 'step    1/10' in the file.")
                أرجع ڪجهه_نه

            # Read 10 rows starting from the identified start row
            لكل سٽ في سٽون[شروعات_اشاريو:شروعات_اشاريو + 10]:
                # Extracting the specified columns
                عدد = عائم(سٽ[کانو_شروعات:کانو_پڇاڙي].strip())
                numbers.append(عدد)
        أرجع numbers
    استثناء Exception كـ e:
        ڇاپيو(f"Error reading the file: {e}")
        أرجع ڪجهه_نه

وصف compare_numbers(read_قدرون, fixed_قدرون, percent_accuracy):
    لكل i في سلسلو(ڊيگهه(read_قدرون)):
        read_قدر = read_قدرون[i]
        fixed_قدر = fixed_قدرون[i]
        percent_difference = ((read_قدر - fixed_قدر) / fixed_قدر) * 100
        ڇاپيو(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        إذا abs(percent_difference) > percent_accuracy:
            ڇاپيو(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            أرجع 1
    ڇاپيو("Success: All values are within the allowed accuracy.")
    أرجع 0

وصف مکيه():
    parser = argparse.دليل_Parser(description='Compare numbers in a file with fixed values.')
    parser.جوڙيو_دليل('-f', '--file', required=سچ, help='Path to the input file')
    parser.جوڙيو_دليل('-s', '--col_start', type=صحيح, required=سچ, help='Starting column index (0-based)')
    parser.جوڙيو_دليل('-e', '--col_end', type=صحيح, required=سچ, help='Ending column index (0-based)')
    parser.جوڙيو_دليل('-a', '--percent_accuracy', type=عائم, required=سچ, help='Allowed percent accuracy for comparison')

    دليلون = parser.parse_دليلون()

    # Read numbers from file
    read_قدرون = read_numbers_from_file(دليلون.file, دليلون.کانو_شروعات, دليلون.کانو_پڇاڙي)
    إذا read_قدرون آهي ڪجهه_نه:
        أرجع 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_قدرون = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    نتيجو = compare_numbers(read_قدرون, fixed_قدرون, دليلون.percent_accuracy)
    أرجع نتيجو

إذا __name__ == "__main__":
    sys.exit(مکيه())
