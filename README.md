### Domain Generating Algorithm based Malicious Domains Detection
Botnets often use Domain Generating Algorithms (DGAs) to facilitate covert server communication in carrying out different types of cyber-attacks. Attackers employ these algorithms to generate millions of sites for victim machines to connect to, thus evading defense using blacklists. DGAs enables attacks to be facilitated without the fear of command and control (C\&C) servers being identified and permanently blocked.

In this study, we develop a suite of effective detection mechanisms for both traditional and dictionary-based DGAs. We analyze 84 different malware families’ dataset with the Virus-Total API service to verify the status of each of the malicious domains using multiple Antivirus scan engines. We utilize DNS queries to determine whether a domain is a non-existent domain (NXDomain) and hence, short-lived. We extract useful features to further analyze the domain strings. The Bigram model was used with several Machine Learning techniques, such as Logistic Regression, Decision Tree, and Artificial Neural Network (ANN), and the Word2Vec model was used with Long Short-Term Memory (LSTM).

The work has been published at the [8th IEEE International Conference on Cyber Security and Cloud Computing (IEEE CSCloud 2021)](http://www.cloud-conf.net/cscloud/2021/cscloud/index.html).

## Citing this work
If you use our implementation for academic research, you are highly encouraged to cite [our paper]().


```
@inproceedings{ayub2021domain,
  title={Domain Generating Algorithm based Malicious Domains Detection},
  author={Ayub, Md Ahsan and Smith, Steven and Siraj, Ambareen and Tinker, Paul},
  booktitle={2021 8th IEEE International Conference on Cyber Security and Cloud Computing (CSCloud)/2021 7th IEEE International Conference on Edge Computing and Scalable Cloud (EdgeCom)},
  year={2021},
  organization={IEEE}
}
```

The work has been jointly funded by the [Cybersecurity Education, Research & Outreach Center (CEROC)](https://www.tntech.edu/ceroc/) and the [College of Engineering (CoE)](https://www.tntech.edu/engineering/) at [Tennessee Tech University](https://www.tntech.edu).
