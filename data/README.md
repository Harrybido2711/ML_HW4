## HW4 Dataset: Speeches

This assignment  will analyze a dataset of the text from State of the Union
speeches. To save disk space on the autograder, you need to download those into
your repository from Canvas.  The `.gitignore` file will by default prevent
those files from being pushed to GitHub; please do not override that. When we
run the autograder we will copy the dataset over into your repo.

We've uploaded the dataset to [the Canvas 'Homework Handouts' Files folder](
https://canvas.northwestern.edu/courses/252410/files/folder/Homework%20Handouts)
as both a tgz archive and a zip archive. Try downloading one of those two from
there, and then copy it into your repository in the data/ folder. Once it’s
there, you’ll need to extract the archive. Depending on your computer and
operating system, you should have at least one of zip or tar installed; use the
corresponding file. 

  - If you have the tgz file, run:
    - `tar xzvf speeches.tgz`

  - If you have the zip file, run:
    - `unzip speeches.zip`

  - If neither command works and you have a Windows machine, you can download
    [7zip](https://7-zip.org/download.html).

  - This should create a openml folder such that your repository structure
    looks something like:

    ```
    ├─ data/
    │   ├─ README.md        # this file
    │   ├─ stopwords.txt
    │   └─ speeches/
    │       ├─ 1790_Washington
    │       ├─ 1791_Washington
    │       └─ etc.
    ├─ src/
    │   ├─ __init__.py
    │   └─ naive_bayes.py
    ├─ free_response/
    │   ├─ q1.py
    │   └─ data.py
    ```

Please leave the `data.py` file where it is inside of `free_response/`.  If you
have any issues, please ask for help.
