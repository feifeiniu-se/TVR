Zero-Shot Prompt:

```
Please check if the message or signal from the stakeholder requirement is correctly covered by the system requirement.
Please focus only on verifying the message or signal mentioned, without considering other parts of the requirement.
Stakeholder Requirement: <stakeholder>{stakeReq}</stakeholder>
System requirement: <system>{sysReq}</system>
Only respond with either "Yes" or "No":
```

Chain-of-Shot Prompt:

```
Please check if the message or signal from the stakeholder requirement is correctly covered by the system requirement.
Please focus only on verifying the message or signal mentioned, without considering other parts of the requirement.
Stakeholder Requirement: <stakeholder>{stakeReq}</stakeholder>
System requirement: <system>{sysReq}</system>
Let's think step by step and only respond with either "Yes" or "No":
```

Few-Shot Prompt: 

```
Please check if the message or signal from the stakeholder requirement is correctly covered by the system requirement.
Please focus only on verifying the message or signal mentioned, without considering other parts of the requirement.
Example: <example>{examples}</example>
Now evaluate the following step by step and only respond with either "Yes" or "No":
Stakeholder Requirement: <stakeholder>{stakeReq}</stakeholder>
System requirement: <system>{sysReq}</system>
Response:
```

RAG Prompt:

```
Please check if the message or signal from the stakeholder requirement is correctly covered by the system requirement.
Please focus only on verifying the message or signal mentioned, without considering other parts of the requirement.
Example: <example>{examples}</example>
Now evaluate the following step by step and only respond with either "Yes" or "No":
Stakeholder Requirement: <stakeholder>{stakeReq}</stakeholder>
System requirement: <system>{sysReq}</system>
Response:
```

Please note that the prompts for few-shot and RAG are identical, except for the examples. In the few-shot prompt, `{examples}` are randomly selected from the dataset and remain the same for all predictions. In contrast, in RAG, `{examples}` are dynamically retrieved from the database based on their similarity to the predicted pairs.
