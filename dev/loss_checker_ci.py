# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
దిగుమతి sys
దిగుమతి argparse

నిర్వచనం read_numbers_from_file(file_మార్గం, నిలువు_ప్రారంభం, నిలువు_చివర):
    ప్రయత్నించు:
        numbers = []
        తోడుగా open(file_మార్గం, 'r') ఇవ file:
            పంక్తులు = file.readlines()
            ప్రారంభం_సూచీ = ఏమీలేదు
            కొరకు i, పంక్తి లో enumerate(పంక్తులు):
                ఒకవేళ "step    1/10" లో పంక్తి:
                    ప్రారంభం_సూచీ = i
                    ఆపు

            ఒకవేళ ప్రారంభం_సూచీ అయిన ఏమీలేదు:
                ముద్రించు("Error: Could not find the string 'step    1/10' in the file.")
                తిరిగి ఏమీలేదు

            # Read 10 rows starting from the identified start row
            కొరకు పంక్తి లో పంక్తులు[ప్రారంభం_సూచీ:ప్రారంభం_సూచీ + 10]:
                # Extracting the specified columns
                సంఖ్య = దశాంశం(పంక్తి[నిలువు_ప్రారంభం:నిలువు_చివర].strip())
                numbers.append(సంఖ్య)
        తిరిగి numbers
    మినహాయింపు Exception ఇవ e:
        ముద్రించు(f"Error reading the file: {e}")
        తిరిగి ఏమీలేదు

నిర్వచనం compare_numbers(read_విలువలు, fixed_విలువలు, percent_accuracy):
    కొరకు i లో శ్రేణి(పొడవు(read_విలువలు)):
        read_విలువ = read_విలువలు[i]
        fixed_విలువ = fixed_విలువలు[i]
        percent_difference = ((read_విలువ - fixed_విలువ) / fixed_విలువ) * 100
        ముద్రించు(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        ఒకవేళ abs(percent_difference) > percent_accuracy:
            ముద్రించు(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            తిరిగి 1
    ముద్రించు("Success: All values are within the allowed accuracy.")
    తిరిగి 0

నిర్వచనం ప్రధాన():
    parser = argparse.వాదం_Parser(description='Compare numbers in a file with fixed values.')
    parser.కూడు_వాదం('-f', '--file', required=నిజం, help='Path to the input file')
    parser.కూడు_వాదం('-s', '--col_start', type=పూర్ణాంకం, required=నిజం, help='Starting column index (0-based)')
    parser.కూడు_వాదం('-e', '--col_end', type=పూర్ణాంకం, required=నిజం, help='Ending column index (0-based)')
    parser.కూడు_వాదం('-a', '--percent_accuracy', type=దశాంశం, required=నిజం, help='Allowed percent accuracy for comparison')

    వాదనలు = parser.parse_వాదనలు()

    # Read numbers from file
    read_విలువలు = read_numbers_from_file(వాదనలు.file, వాదనలు.నిలువు_ప్రారంభం, వాదనలు.నిలువు_చివర)
    ఒకవేళ read_విలువలు అయిన ఏమీలేదు:
        తిరిగి 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_విలువలు = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    ఫలితం = compare_numbers(read_విలువలు, fixed_విలువలు, వాదనలు.percent_accuracy)
    తిరిగి ఫలితం

ఒకవేళ __name__ == "__main__":
    sys.exit(ప్రధాన())
