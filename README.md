# RootFinding

Root finding methods for multivariate polynomial equations.
Developed at Brigham Young University 2016-2018.

## Getting Started

Welcome!
Writing code as a team can be a challenge.
The rest of this README will show you how to contribute to this project in a way that keeps things organized and working.

**Brief Summary of Commands**

- `git branch`: Shows you which branch you are currently working on.
- `git checkout branchname`: Switch your working branch to the branch called "branchname".
- `git checkout -b branchname frombranch`: Create a new branch called "branchname" that is based off of the code in the branch called "frombranch".
- `git add filename`: Tells git to keep track of the changes you make to a file called "filename" on your local machine.
- `git commit -m "descriptive message"`: Stores the changes you have made to files that are being tracked on your local machine along with a message describing what changes you made to the files.
- `git push origin branchname`: Send the changes made on the current branch to github.com to be stored on the repository website under the branch called "branchname".
- `git pull origin branchname`: Grabs the most recent version of the branch called "branchname" from github.com and updates your local machine with any changes that exist in the online branch.
- `git branch -D branchname`: Force git to delete the branch called "branchname" from your local machine.

**Issues**

When you are assigned to write new code or modify existing code, you or someone on the team should create an issue on this GitHub page by clicking the new issue button at the top of this page.
Give the issue a specific title that describes your assignment.
Assign yourself to the issue and save it.
This helps us keep track of who is responsible for which assignments.
When you have finished resolving the issue (meaning your code has been successfully merged into develop) you can close the issue.

**Test-Driven Development**

When you are given an assignment, you should first write all the unit tests for the code you are about to write.
Assuming the unit tests are correct, all you have to do is write or modify code until all the unit tests pass.
This is called *test-driven development* and helps you avoid writing unnecessary or complicated code that doesn't directly contribute to the solution you intend to build.
Writing the unit tests first also helps you fully understand the problem you are trying to solve before you code up the solution.
This will help you avoid lots of errors that may go unnoticed for a long time.
Save yourself the headache of debugging errors in your code by writing the unit tests first.

**An Example for the Uninitiated**

First, clone the repository from GitHub using the terminal or shell:

`$ git clone https://github.com/tylerjarvis/RootFinding.git`

This repo uses two main branches.
The most stable version of the code is in the branch `master`.
The current code under development is in the branch `develop`.
The `develop` branch will eventually be merged into the `master` branch, signifying that a new version of the package is available to use. You can see which branches are on your local machine and which branch you are currently on by running the following command in the terminal:

`$ git branch`

Suppose that at the last research meeting, you were assigned to test code that converts polynomials in the power basis to polynomials in the Chebyshev basis.
Someone will have created an issue on the GitHub page and assigned you to the issue.
It might be called something like "Write a unit test to check converting polynomials from the power basis to the Chebyshev basis."
You need to create a new branch off of the `develop` branch where you can safely make changes to the code.
Switch to the `develop` branch, then make sure it is up-to-date with what is currently on GitHub:

`$ git checkout develop`

`$ git pull origin develop`.

Now you are ready to create a new branch and switch to it.
You are supposed to a unit test that checks whether some code successfully converts polynomials between bases.
Create a new branch off of develop called `test/convert_poly`.

`$ git checkout -b test/convert_poly develop`

As you modify the code, you should save your progress frequently by committing your code.
You should commit your code anytime you make progress, which usually means a few times per hour.

```
$ git add test_convert.py
$ git commit -m "Wrote the first unit test to check polynomial basis conversion."
```

When you are done working, commit and push your branch to GitHub.

```
$ git add test_convert.py
$ git commit -m "Finished writing the third unit test for polynomial basis conversion."
$ git push origin test/convert_poly
```

**Submitting a pull request**

When you are finished with the assignment, your branch should be ready to merge into develop.
Run `py.test` from the main RootFinding directory to make sure the code still passes the unit tests.

`$ py.test`

Next, commit and push your code.

```
$ git add test_convert.py
$ git commit -m "Polynomial basis conversion passes all 5 of the new unit tests."
$ git push origin test/convert_poly
```

Your code is now visible on the GitHub page online.
You can submit a pull request on the GitHub page.
Although you can merge your own pull request into `develop`, it is best practice to have someone else complete the merge.
Do not complete a merge if there are conflicts or if the code does not have proper documentation, including docstrings and comments.
As soon as your pull request has been successfully merged into develop, you should delete the branch on your local machine.

`$ git branch -D test/convert_poly`

The person who completed the pull request should also delete the branch located on GitHub.
