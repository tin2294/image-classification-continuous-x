# Deploying machine learning systems 

Congratulations! You've been offered an exciting and lucrative new position. You're going to be a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food.

On your first day at work, you are given the following brief:

> When someone uploads a photo of food to our site, it should be tagged and categorized automatically, right?
> 
> Your predecessor had already implemented a machine learning model that classified photos into different categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. And they built a simple web app around the upload and classification process, as well as a deployment "recipe" for the web app.
> 
> But, there were problems: one, the categorization was not very accurate. And second, it took too long. The classification happens in real time, when the photo is uploaded and before the user sees the next page (with the photo and info about it, including its assigned category), so it can't take too long.
> 
> Of course, we can fix the timing issue to some extent by throwing more compute resources at the problem. But we are also concerned with cost, so we don't want to pay for compute resources that are sitting idle when load is low.
>
> You can fix both of these problems, right? (That's why we hired you!)

You are expected to deliver the following:

* an evaluation of your predecessor's model and deployment "recipe" (both in terms of optimizing metrics like latency, and operational metrics like inference time and resource usage.)
* a new model and deployment "recipe" to optimize the accuracy of the model. (Of course, you don't want to sacrifice on operational metrics unnecessarily - don't make choices that increase inference time or resource usage but don't improve accuracy. But, when there is a tradeoff between accuracy and latency, prioritize accuracy.)
* a new model and deployment "recipe" to optimize the inference latency of the model. (Of course, you don't want to sacrifice on accuracy unnecessarily - don't make choices that reduce accuracy but don't improve inference time or resource usage. But, when there is a tradeoff between latency and accuracy, prioritize latency.)
* a PDF set of slides that your manager can use to present your overall findings to the senior executives at GourmetGram.

