#RootFinding

Root finding methods for multivariate polynomial equations.

**For Contributors**

***Git Workflow***

To contribute to this repository, clone the repository from github using the terminal or shell:

`git clone https://github.com/tylerjarvis/RootFinding.git`

***Branches***
This repo uses two main branches.
The most stable version of the code is in the branch `master`.
The current code under development is in the branch `develop`.
Side note: After the develop branch is stable with significant changes, the develop branch will be merged into the master branch, creating a new master branch. This represents upgrading the master branch to a new version (ie version 1.0 to version 2.0).

When you are contributing, you can see which branch you are on by running the following command in the terminal:

`git branch`

***Creating a new branch to work on***
Before creating a new branch, switch to the develop branch by typing

`git checkout develop`.

Make sure it is up-to-date by typing

`git pull origin develop`.

Now you are ready to create a new branch and switch to it.
To do this type

`git checkout -b branchname`

where `branchname` refers to a descriptive name of what you are working on.

For example, if your job is to write code that converts power basis polynomials to chebyshev basis polynomials, you could run

`git checkout -b feature/polynomial_converter`

which creates a new branch called `feature/polynomial_converter`.
This is good practice since it tells anyone else who sees this branch that you are developing a new feature.

If you wanted to update monomial multiplication in a file called `multi_cheb.py`, you could run

`git checkout -b feature/monomial_multiplication develop`

to create a new branch called `feature/monomial_multiplication`.
This is good practice because it tells anyone else who sees this branch that you are changing old code based off of the develop branch.

***Saving progress by pushing to github***
To save your progress at the end of a work period (probably good to do hourly or so) you should commit your changes and push your branch up to github by doing the following (according to the above example)

```
git add multi_cheb.py
git commit -m "Fixed some errors in the monomial multiplication code. Comments and documentation is not yet written."
git push origin feature/monomial_multiplication
```

***Submitting a pull request***
When you think your branch is ready to merge into develop, first make sure everything still passes the unit tests by typing
`py.test`
in the main RootFinding directory.
If all the tests pass, make sure someone else looks over your code.
Make sure they understand what your code is doing and why the changes are needed.
The reviewer should make sure that you have written good docstrings and have commented your code so that it is easy to follow.
After making any last changes to the code (and making sure it still passes the unit tests), you can push your code directly to github by typing

`git push origin feature/monomial_multiplication`.
This pushes your branch to github and can be visible on the github page online.

You can then submit a pull request to merge your branch into the develop branch.
Although you can merge your own pull request, it is best practice to have someone else to approve and merge in your pull request.

**Organization of this Repository**


**Solving a System of Polynomial Equations**

Groebner bases can be used to find solutions to a system of polynomial equations.
Unfortunately, computing Groebner bases is known to be unstable due to floating point error propagation.
Polynomials in the power basis can be represented by a summation of Chebyshev polynomials, which are good for stable computation.
Converting the polynomials to Chebyshev polynomials reduces the error significantly.
Here's an example using the code from this repository:

*Example*

Developed at Brigham Young University 2016-2017
