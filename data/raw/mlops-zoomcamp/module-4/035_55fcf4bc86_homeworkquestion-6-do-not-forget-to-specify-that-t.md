---
id: 55fcf4bc86
question: 'Homework/Question 6: Do not forget to specify that the folder output/yellow
  should be created in the working directory of your docker file'
sort_order: 35
---

For question 6, which requires you to include your script in a Dockerfile, specify the creation of the folder `output/yellow` in the working directory of your Docker container by adding the following line in your Dockerfile:

```Dockerfile
RUN mkdir -p output/yellow
```
