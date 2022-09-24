import numpy as np

class F1:
    def compute_score(self, gts, res):
        """
        Main function to compute F1 score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: f1 (float) : computed F1 score for the corpus
        """
        res = {key: value[0].split() for key, value in res.items()}
        scores = []
        for key in res:
            r = res[key]
            gt = gts[key]
            
            # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
            if len(r) == 0 or len(gt) == 0:
                scores.append(int(r == gt))
            else:
                common_tokens = set(r) & set(gt)
                # if there are no common tokens then f1 = 0
                if len(common_tokens) == 0:
                    scores.append(0)
                else:
                    prec = len(common_tokens) / len(r)
                    rec = len(common_tokens) / len(gt)

                    scores.append(2*(prec*rec)/(prec+rec))

        scores = np.array(scores)

        return scores.mean(), scores

    def __str__(self) -> str:
        return "F1"
