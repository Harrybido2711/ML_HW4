# CS 349 HW 4: Naive Bayes and the EM Algorithm

There are 23 points possible for this assignment. 1 point is for the setup, 16
points are for the code, and 6 points are for the free-response questions. The
setup portion is due earlier than the other pieces -- all deadlines are on
Canvas.

**Please carefully read this entire README before starting the assignment.**

You should check the `naive_bayes.pdf` write-up before beginning your
implementation. And you can check the `debugging_hints.pdf` for some helpful
hints.

## What's changed since HW3?

- You need to download the dataset we'll use from Canvas. See the `README.md`
  file in the `data/` directory for more information.
- We've moved the `data.py` file into `free_response/`. You won't need to edit
  that file.
- To pass any tests locally (including the setup), you will need to download
  the dataset first. However, you can still pass the setup test on the
  autograder without downloading the dataset.
- Please make sure you are using a `scipy` version >= 1.15. This will likely
  save you some headaches in the long run.

## Academic integrity

Your work must be your own. You may not work with others. Do not submit other
people's work as your own, and do not allow others to submit your work as
theirs. 

If you need help debugging your code, make a *private* post on Piazza or come
to office hours. You may not show your code (including pseudocode) to other
students under any circumstances.

You are required to completely understand any homework solution that you
submit, and, in case of any doubt, you must be prepared to orally explain your
solution. If you have submitted a solution that you cannot verbally explain,
then you have violated this policy.

Please see [the academic integrity
policy](https://canvas.northwestern.edu/courses/252410/pages/academic-integrity)
for more detail.  By pushing your code to GitHub, you agree to these rules, and
understand that there may be severe consequences for violating them.

## Important instructions

Your work will be graded and aggregated using an autograder that will download
the code and free response questions from each student's repository.

The essential instructions:
- We will only grade the latest version of your code that was pushed to GitHub
  before the deadline (accounting for late days; see below).
- Your NetID must be in the `netid` file; replace `NETID_GOES_HERE` with your
  netid.
- Your answer to each free response question should be upload to Canvas in *its
  own PDF* with the filename `qYYY.pdf`, where `YYY` is the question number.
- Do not include your name or Net ID in the content of your free response PDFs
  -- we will grade these anonymously.

## Late Work

In general, unexcused late work is worth zero points. The autograder will only
download work from your repository that was pushed to GitHub before the
deadline. However:

- Each student gets four late days to use across the entire quarter. If you
  want to use late days, use the [late day assignment
  ](https://canvas.northwestern.edu/courses/252410/assignments/1757674) on
  Canvas.
- You can use at most two late days per assignment.
- If you have a personal emergency, please ask for help. You do not have to
  share any personal information with me, but I will ask you to get in touch
  with the dean who oversees your student services to coordinate
  accommodations.

## Clone this repository and environment setup

If you are using `conda`, you can just use the same environment for this
assignment that you used for HW1.  If you are using `uv`, you may need to
create a new environment for each assignment individually. You should be able
to just run `uv sync` from the root directory of this repository.  For more
detailed versions of these instructions, refer to the HW1 README.

## What to do for this assignment

The detailed instructions for the work you need to do are in `problems.md`.
You will also find it very helpful to read the `naive_bayes.pdf` writeup
we share in this repo.

For the coding portion of the assignment, you will:
- Solve some simple practice problems with sparse matrices
- Write stable softmax and log sum functions
- Implement a fully-supervised NaiveBayes classifier
- Implement a semi-supervised NaiveBayes classifier

You will also write up answers to the free response questions.

In every function where you need to write code, there is a `raise
NotImplementedError` in the code. The test cases will guide you through the work
you need to do and tell you how many points you've earned. The test cases can
be run from the root directory of this repository with:

``python -m pytest``

To run a single test, you can call e.g., `python -m pytest -s -k test_setup`.
The `-s` means that any print statements you include will in fact be printed;
the default behavior (`python -m pytest`) will suppress everything but the
pytest output.

We will use these test cases to grade your work! Even if you change the test
cases such that you pass the tests on your computer, we're still going to use
the original test cases to grade your assignment.

## Questions? Problems? Issues?

Simply post on Piazza, and we'll get back to you.
