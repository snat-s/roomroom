This has currently two folders, one has Binoculars a method to detect AI
generated text and the other one a directory called `roomroom`. This second
folder has a PrimeIntellect style of enviorment where we train a really small
model, in the style of a Qwen model that is trained to pass as a non AI model.

Why do this? Because I wanted to see a model that could write more like a human
instead of an AI. And this is the result of that experiment.

On an instance I have created another file where I explaine exactly how
Binoculars works. Its called `BINOCULARS_GUIDE.md`. You can read it fully to
understand how it works.

I think that what I want to do for now is as a first step, run Binoculars with
two different models than the falcon ones. I want to use the smallest of Qwen3s
family. They are really small and nice models. Their names are the following
`Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-0.5B-Instruct`.

I have copied all of the Binoculars into the folder `binoculars/` right in this
folder.

The first task I want to do is add all the needed dependencies to run this
with no other dependency and I want to install appropiate dependencies using
`uv`.

I have a dataset that is inside of the `data` folder. There is a .csv called
`data/ai_vs_human_text.csv`.

So that we can calibrate the models, because right now it uses the thresholds
for the other dataset. jionghong94/GhostBuster_v3 and a bit more for ccdv/cnn_dailymail 
is it biased for english? yes. 
