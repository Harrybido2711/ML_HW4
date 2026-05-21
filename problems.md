## Instructions

There are 23 points possible for this assignment. 1 point is for the setup, 16
points are for the code, and 6 points are for the free-response questions. The
setup portion is due earlier than the other pieces -- all deadlines are on
Canvas. There is no mini-project for this assignment.

### Setup (1 point)

All you need to do for these points is pass the test_setup case. This requires
putting your NetID in the netid file. If you delete your NetID or if your code
has syntax errors, the autograder may give you a zero. If you think the
autograder has unfairly given you fewer points than you should have earned, it
is your responsibility to let us know. If mistakes on your end require us to
manually grade your code, we will deduct points.

# Coding (16 points)

Start by solving the practice problems in `src/sparse_practice.py`; these
will help you understand how scipy sparse matrices work.

In `src/utils.py`, you will write `softmax` and `stable_log_sum` functions,
making sure that both are numerically stable. These will be helpful in your
Naive Bayes models.

Then, you will implement two versions of a Naive Bayes classifier.  In
`src/naive_bayes.py`, the `NaiveBayes` classifier considers the case where all
the data is labeled.  In `src/naive_bayes_em.py`, `NaiveBayesEM` classifier
will use the Expectation Maximization algorithm to also learn from unlabeled
data.

The lecture slides on Canvas and the PDF write-up in this repo will
be helpful.  We have also provided extensive documentation in the provided
code, please read it carefully!  For example, when implementing the
`NaiveBayesEM` classifier, be careful to correctly initialize your parameters
and correctly update your inferred class distribution during the E-step (i.e.,
do not overwrite the observed labels with your predicted probabilities).

Your goal is to pass the test suite that is run by `python -m pytest`.
Once the tests are passed, move on to the free-response questions below.
The tests build on and sometimes depend on each other. We suggest that you
implement them in the order they appear in `tests/rubric.json`. That file also
allows you to see how many (relative) points each test is worth and which other
tests it may depend on. 

You may not use `sklearn` or `scipy` to implement the functions in this
assignment. Please do not use the python internal modules or functions
`importlib`, `getattr`, or `globals`. The `test_imports` case will try to alert
you if you use this disallowed packages or functions; please do not try to
circumvent these checks. If you think the test case is erroneously penalizing
you, please make a private Piazza post.
 
The grade given to you by the autograder on Canvas is the grade you should
expect receive. If something goes wrong (your code times out, you import a
disallowed package, you accidentally push a syntax error, etc.) and you need us
to grade your code manually, we will do so but subtract a 2 point penalty.
Please be careful and read the feedback that the autograder is giving you.

## The speeches dataset

The dataset provided in `data/speeches.zip` (which should be automatically
unzipped by the `free_response/data.py` code) contains [State of the Union
addresses](https://en.m.wikisource.org/wiki/Portal:State_of_the_Union_Speeches_by_United_States_Presidents)
by United States presidents dating back to 1790. In recent history
(specifically, since the [Civil Rights Act of
1964](https://simple.wikipedia.org/wiki/Party_realignment_in_the_United_States#1960s-80s)),
all US presidents have belonged to one of two political parties, which have
remained relatively ideologically stable.

Please see `data/README.md` for a guide to getting set up with this dataset.

In this data, we treat the words of each speech as the features, and the
political party of the speaker as the label.  For presidents prior to 1964, we
will treat their party membership as unobserved, as it does not necessarily
correspond to today's two-party system. The `NaiveBayes` classifier will only
consider fully-labeled data -- it cannot use speeches prior to 1964. The
`NaiveBayesEM` classifier will also use unlabeled speeches to learn its
probabilities.

# Free-response questions (6 points)

As before, each question should be answered in its own PDF file. We will grade
these anonymously; please do not include your name, NetID, or other
personally-identifying information. We will deduct points for this if
necessary.

## 1. Comparing Naive Bayes with and without unlabeled data (1 point)

Try running `python -m free_response.q1`. The code creates a dataset using
`src.data.build_dataset` with 100 documents, at most 2000 words per document,
and a vocabulary size of 1000 words. As above, we will consider speeches prior
to 1964 as unlabeled. The code fits your `NaiveBayes` and `NaiveBayesEM` models
on the `data` matrix and `labels` array. For the `NaiveBayesEM` model, it
trains with `max_iter` values of 1, 2, and 10.

If your implementation is correct, we expect it to produce values *similar but
not exactly equal* to the values below. You can use these values to check the
approximate correctness of your code. If you complete the HW4 coding assignment,
please use **your exact values** rather than those shown below, and **please
include those values** in your answer.

If you cannot complete the coding assignment, please use the values provided
below to answer this question.

| Model | `max_iter` | Accuracy | Log Likelihood |
| ---   | ---        | ---      | ---            |
| NB    | N/A        | 98.1%    | -31217.4       |
| NB+EM | 1          | 100%     | -57930.3       |
| NB+EM | 2          | 95.4%    | -56878.9       |
| NB+EM | 10         | 76.7%    | -55296.4       |

Consider the differences in accuracy and likelihood between the `NaiveBayes`
and `NaiveBayesEM` models. Why do the `NaiveBayesEM` models have a lower
likelihood than the `NaiveBayes` model? Why is accuracy not always positively
correlated with likelihood?

## 2. Interpreting `beta` values (1 point)

Try running `python -m free_response.q2`. The code is similar to that of
`q1.py` above; it trains a `NaiveBayesEM` model on a similar dataset.  After
training, it extracts some specific values of the model's learned `beta` and 
plots them in a table. 

If your implementation is correct, we expect it to produce values *similar but
not exactly equal* to the values below. You can use these values to check the
approximate correctness of your code. If you complete the HW4 coding assignment,
please use **your exact values** rather than those shown below, and **please
include those values** in your answer.

If you cannot complete the coding assignment, please use the values provided
below to answer this question.

|Word       |  Beta[:, 0] | Beta[:, 1] |
| ---       | ---         | ---        |
|terror     |  0.219      | 0.044      |
|strategic  |  0.541      | 0.122      |
|allies     |  0.643      | 0.122      |
|banks      |  0.054      | 0.313      |
|loans      |  0.129      | 0.601      |
|ought      |  0.078      | 0.313      |
|set        |  0.642      | 0.642      |
|policies   |  0.642      | 0.642      |
|expect     |  0.358      | 0.358      |

While you don't need to modify the code, you will need to look at `q2.py` file
to answer this question.

In your own words, what is this code doing? Please discuss the `f(word)`
function (`f_scores` in the code): how do we calculate, interpret, and sort it?
What does the table above say about words like "banks," "terror," and "set"?
Please interpret these words as `partisan` (`Democratic` or `Republican`) or
`neutral`.

## 3. Probabilistic predictions (2 points)

Try running `python -m free_response.q3`. The code is similar to that of
`q1.py` and `q2.py` above; it trains a `NaiveBayesEM` model on a similar
dataset. After training, it uses the `predict_proba` function to predict the
probability that each labeled speech in the training data is from a Democratic
or Republican president. We can think about these probabilities as describing
the "confidence" of the classifier -- if it outputs a probability of 50% for
both labels, the model is unconfident. When the model's probability of the
predicted label is close to 100%, it is very confident in its prediction.

If your implementation is correct, we expect it to produce values *similar but
not exactly equal* to the values below. You can use these values to check the
approximate correctness of your code. If you complete the HW4 coding assignment,
please use **your exact values** rather than those shown below, and **please
include those values** in your answer.

If you cannot complete the coding assignment, please use the values provided
below to answer this question.

```
Across 55 correct predictions,  NBEM has average confidence 99.14%
Across 3 incorrect predictions, NBEM has average confidence 99.41%
```

(a) What is one aspect of the data and/or model that could contribute to the
patterns you see in these results? Justify your answer as best you can.

(b) Suppose we were using a machine learning classifier in a high-stakes domain
such as classifying whether images of tumors are benign or malignant. What
might be one danger of having a classifier that is always confident, even when
incorrect?

(c) In the results above, out of 58 predictions the model was correct less than
99% of the time, but its average confidence was above 99%. If the [model were
calibrated](https://scikit-learn.org/stable/modules/calibration.html), a
prediction with confidence of X% would be correct X% of the time. Explain in
one sentence: what would be an important benefit of a calibrated classifier,
where its confidence represents the probability that it makes a correct
prediction?

(d) Suppose you wanted to introduce a form of regularization to make our
NaiveBayesEM model less overconfident in its predictions. To do this, you might
modify how we use `smoothing` or use a prior probability to guide how we
learn `beta`. Propose a method for doing this and justify why this would work.

## 4. Fairness metrics revisited (1 point)

(a) Go back to the HW 1 materials and look at the definitions for Equalized
Odds and Demographic Parity. Rephrase these definitions as statements about
(conditional) independence of variables (e.g., `protected group` and `model prediction`).

(b) 2NB is a Naive Bayes method modified to make fairer classifications,
described [in this paper](
https://link.springer.com/article/10.1007/s10618-010-0190-x).
It uses demographic parity (also known as statistical parity) to measure
fairness. Suppose you use a 2NB model to diagnose whether or not a
patient has sickle cell anemia (a blood disorder which is [especially common
among Black Americans](https://www.cdc.gov/sickle-cell/data/)).
What is one issue that could arise from using demographic parity to ensure
this diagnoses are fair? Justify your answer.

(c) What is one other way you could try to make your model fairer in these
predictions? Explain why this would be a good choice.

## 5. Bayesian Improved Surname Geocoding (1 point)

Skim through [this paper published in Science in
2022](https://www.science.org/doi/10.1126/sciadv.adc9824): ``Addressing census
data problems in race imputation via fully Bayesian Improved Surname Geocoding
and name supplements.'' The authors introduce an extension to a typical Naive
Bayes classifier, which introduces modeling challenges beyond what we covered
in class.

(a) What is one challenge or methodological approach that the authors discuss
that we also covered in lecture (or elsewhere in this assignment)? Explain how
their discussion aligns with or differs from our coverage.

(b) What is one challenge or methodological approach that the authors discuss
that we did not cover in lecture? In your own words, explain why challenge or
approach this matters. Don't quote from the original article.

(c) Reflect on the ethical questions raised by this work. Why are the risks of
using this classifier for studies of racial of disparities? Do you think the
potential benefits are justified despite those risks? Why or why not?
