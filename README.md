# Supervised Morphological Segmentation using Conditional Random Fields

Implementation of the paper: [Supervised Morphological Segmentation in a
Low-Resource Learning Setting using Conditional Random Fields](https://www.semanticscholar.org/paper/Supervised-Morphological-Segmentation-in-a-Learning-Ruokolainen-Kohonen/b83810c620671ecadacafad41e7bce370284bcfe), CoNLL 2013

Homework for the course of Natural Language Processing (a.y. 2016/2017) at Sapienza, University of Rome. 

## Summary 

Idea: Treat morphological segmentation as a classification problem of a
sequence of letters.

Tagset: 
```
START
B      # begin
M      # middle
E      # end
S      # singleton
STOP
```
e.g. the segmentation of the word _drivers_ is:
```
<w>    d r i v + e r + s </w>
START  B M M E   B E   S END
```
Where `<w>` and `</w>` are additional start and end markers.

## Usage
To run the experiment, let say, `1`, go into the root folder and run:
```
python main.py --demo 1 --verbosity
```

You can run 3 demos that I've set up.
- demo 1: Train the model using part of the entire training set (e.g. 10%, 20%...) and compare the results.
- demo 2: do a grid search to find the best configuration of parameters:
    - `delta` (max length of substrings near to the current char), 
    - `max_iterations` and `epsilon` of the perceptron algorithm.
- demo 3: the same of demo 2, but over a larger dataset.

In the folder `output/demo_X` you'll find more details about the result of the experiment.

## Configurations

You can change some configuration in the file `config.py`;
in particular, you can toggle the flag `LIGHTWEIGHT_SETTING_FLAG` to execute the grid search
in a smaller parameter space (you gain a lot of time...).

I strongly suggest you to use the option `--verbosity`, in order to allow you to figure out
what is happening. After the execution of each demo, you will find
the output produced in the output/demo_xx folder, and a report.txt which
summarize the obtained results and the demo_xx.log file.

