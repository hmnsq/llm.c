# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
ਆਯਾਤ sys
ਆਯਾਤ argparse

ਪਰਿਭਾਸ਼ਾ read_numbers_from_file(file_ਰਸਤਾ, ਖਾਨਾ_ਸ਼ੁਰੂ, ਖਾਨਾ_ਸਿਰਾ):
    ਕੋਸ਼ਿਸ਼:
        numbers = []
        ਨਾਲ open(file_ਰਸਤਾ, 'r') ਇਵ file:
            ਸਤਰਾਂ = file.readlines()
            ਸ਼ੁਰੂ_ਸੂਚਕਾਂਕ = ਕੁਝ_ਨਹੀਂ
            ਲਈ i, ਸਤਰ ਵਿੱਚ enumerate(ਸਤਰਾਂ):
                ਜੇ "step    1/10" ਵਿੱਚ ਸਤਰ:
                    ਸ਼ੁਰੂ_ਸੂਚਕਾਂਕ = i
                    ਰੋਕੋ

            ਜੇ ਸ਼ੁਰੂ_ਸੂਚਕਾਂਕ ਹੈ ਕੁਝ_ਨਹੀਂ:
                ਛਾਪੋ("Error: Could not find the string 'step    1/10' in the file.")
                ਵਾਪਸ ਕੁਝ_ਨਹੀਂ

            # Read 10 rows starting from the identified start row
            ਲਈ ਸਤਰ ਵਿੱਚ ਸਤਰਾਂ[ਸ਼ੁਰੂ_ਸੂਚਕਾਂਕ:ਸ਼ੁਰੂ_ਸੂਚਕਾਂਕ + 10]:
                # Extracting the specified columns
                ਸੰਖਿਆ = ਦਸ਼ਮਲਵ(ਸਤਰ[ਖਾਨਾ_ਸ਼ੁਰੂ:ਖਾਨਾ_ਸਿਰਾ].strip())
                numbers.append(ਸੰਖਿਆ)
        ਵਾਪਸ numbers
    ਅਪਵਾਦ Exception ਇਵ e:
        ਛਾਪੋ(f"Error reading the file: {e}")
        ਵਾਪਸ ਕੁਝ_ਨਹੀਂ

ਪਰਿਭਾਸ਼ਾ compare_numbers(read_ਮੁੱਲਾਂ, fixed_ਮੁੱਲਾਂ, percent_accuracy):
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(ਲੰਬਾਈ(read_ਮੁੱਲਾਂ)):
        read_ਮੁੱਲ = read_ਮੁੱਲਾਂ[i]
        fixed_ਮੁੱਲ = fixed_ਮੁੱਲਾਂ[i]
        percent_difference = ((read_ਮੁੱਲ - fixed_ਮੁੱਲ) / fixed_ਮੁੱਲ) * 100
        ਛਾਪੋ(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        ਜੇ abs(percent_difference) > percent_accuracy:
            ਛਾਪੋ(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            ਵਾਪਸ 1
    ਛਾਪੋ("Success: All values are within the allowed accuracy.")
    ਵਾਪਸ 0

ਪਰਿਭਾਸ਼ਾ ਮੁੱਖ():
    parser = argparse.ਦਲੀਲ_Parser(description='Compare numbers in a file with fixed values.')
    parser.ਜੋੜੋ_ਦਲੀਲ('-f', '--file', required=ਸੱਚ, help='Path to the input file')
    parser.ਜੋੜੋ_ਦਲੀਲ('-s', '--col_start', type=ਪੂਰਨ, required=ਸੱਚ, help='Starting column index (0-based)')
    parser.ਜੋੜੋ_ਦਲੀਲ('-e', '--col_end', type=ਪੂਰਨ, required=ਸੱਚ, help='Ending column index (0-based)')
    parser.ਜੋੜੋ_ਦਲੀਲ('-a', '--percent_accuracy', type=ਦਸ਼ਮਲਵ, required=ਸੱਚ, help='Allowed percent accuracy for comparison')

    ਦਲੀਲਾਂ = parser.parse_ਦਲੀਲਾਂ()

    # Read numbers from file
    read_ਮੁੱਲਾਂ = read_numbers_from_file(ਦਲੀਲਾਂ.file, ਦਲੀਲਾਂ.ਖਾਨਾ_ਸ਼ੁਰੂ, ਦਲੀਲਾਂ.ਖਾਨਾ_ਸਿਰਾ)
    ਜੇ read_ਮੁੱਲਾਂ ਹੈ ਕੁਝ_ਨਹੀਂ:
        ਵਾਪਸ 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_ਮੁੱਲਾਂ = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    ਨਤੀਜਾ = compare_numbers(read_ਮੁੱਲਾਂ, fixed_ਮੁੱਲਾਂ, ਦਲੀਲਾਂ.percent_accuracy)
    ਵਾਪਸ ਨਤੀਜਾ

ਜੇ __name__ == "__main__":
    sys.exit(ਮੁੱਖ())
