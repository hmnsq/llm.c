# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

ಆಮದು json, sys

ಫಲಿತಾಂಶ_2 = sys.argv[1]
ಮುದ್ರಿಸು("-"*40)

ಕೀಲಿ = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

ಒಟ್ಟು = 0
ಗಾಗಿ test ಒಳಗೆ ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    ದತ್ತಾಂಶ = json.loads(open("./%s/%s"%(ಫಲಿತಾಂಶ_2, test)).ಪಠ())
    r_ಎಣಿಕೆ = 0
    r_ಒಟ್ಟು = 0
    ಗಾಗಿ test_name ಒಳಗೆ ದತ್ತಾಂಶ['results']:
      r_ಎಣಿಕೆ += 1
      r_ಒಟ್ಟು += ದತ್ತಾಂಶ['results'][test_name][ಕೀಲಿ[test]]
    score = (r_ಒಟ್ಟು*100)/r_ಎಣಿಕೆ
    ಮುದ್ರಿಸು(f"{test:<30} : {score:.4f}")
    ಒಟ್ಟು += score
ಸರಾಸರಿ_ಮೌಲ್ಯ = ಒಟ್ಟು / 6.0
ಮುದ್ರಿಸು("-"*40)
ಮುದ್ರಿಸು(f"Average Score                  : {average:.4f}")
