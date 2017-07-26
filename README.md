# RootFinding

Root finding methods for multivariate polynomial equations.

## Getting Started

Welcome to the team.
Writing code as a team can be a challenge.
The rest of this README will show you how to contribute to this project in a way that keeps things organized and working.

**An Example**

First, clone the repository from GitHub using the terminal or shell:

`$ git clone https://github.com/tylerjarvis/RootFinding.git`

This repo uses two main branches.
The most stable version of the code is in the branch `master`.
The current code under development is in the branch `develop`.
The `develop` branch will eventually be merged into the `master` branch, signifying that a new version of the package is available to use. You can see which branches are on your local machine and which branch you are currently on by running the following command in the terminal:

`$ git branch`

Suppose that at the last research meeting, you were assigned to test code that converts polynomials in the power basis to polynomials in the Chebyshev basis.
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
Next, commit and push your code.

`$ py.test`

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

**Solving a System of Polynomial Equations**

Groebner bases can be used to find solutions to a system of polynomial equations.
Unfortunately, computing Groebner bases is known to be unstable due to floating point error propagation.
Polynomials in the power basis can be represented by a summation of Chebyshev polynomials, which are good for stable computation.
Converting the polynomials to Chebyshev polynomials reduces the error significantly.
Here's an example using the code from this repository:

*Example*

Developed at Brigham Young University 2016-2017
