# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

ਆਯਾਤ json, sys

ਨਤੀਜਾ_2 = sys.argv[1]
ਛਾਪੋ("-"*40)

ਕੁੰਜੀ = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

ਕੁੱਲ = 0
ਲਈ test ਵਿੱਚ ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    ਅੰਕੜੇ = json.loads(open("./%s/%s"%(ਨਤੀਜਾ_2, test)).ਪਠ())
    r_ਗਿਣਤੀ = 0
    r_ਕੁੱਲ = 0
    ਲਈ test_name ਵਿੱਚ ਅੰਕੜੇ['results']:
      r_ਗਿਣਤੀ += 1
      r_ਕੁੱਲ += ਅੰਕੜੇ['results'][test_name][ਕੁੰਜੀ[test]]
    score = (r_ਕੁੱਲ*100)/r_ਗਿਣਤੀ
    ਛਾਪੋ(f"{test:<30} : {score:.4f}")
    ਕੁੱਲ += score
ਔਸਤਨ = ਕੁੱਲ / 6.0
ਛਾਪੋ("-"*40)
ਛਾਪੋ(f"Average Score                  : {average:.4f}")
