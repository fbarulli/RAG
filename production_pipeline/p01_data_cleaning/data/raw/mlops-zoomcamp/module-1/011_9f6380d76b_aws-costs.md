---
id: 9f6380d76b
question: 'AWS costs:'
sort_order: 11
---

I am worried about the cost of keeping an AWS instance running during the course.

With the instance specified during working environment setup, if you remember to Stop Instance once you finish your work for the day, using that strategy, in a day with about 5 hours of work, you will pay around $0.40 USD, which will account for $12 USD per month. This seems to be an affordable amount.

You must remember that you will have a different public IP address every time you restart your instance, and you will need to edit your SSH Config file. It's worth the time though.

Additionally, AWS enables you to set up an automatic email alert if a predefined budget is exceeded. 

[Here is a tutorial to set this up](https://www.c-sharpcorner.com/article/set-up-an-iam-user-and-the-alert-for-budget-in-aws/).

Also, you can estimate the cost yourself using the [AWS pricing calculator](https://calculator.aws/#/). At the time of writing (20.05.2023), a t3a.xlarge instance with 2 hr/day usage (which translates to 10 hr/week and should be enough to complete the course) and 30GB EBS monthly cost is 10.14 USD.

Here’s [a link to the estimate](https://calculator.aws/#/estimate?id=b24f75864e0af54cbfeeb6083e1c74c605923c65).