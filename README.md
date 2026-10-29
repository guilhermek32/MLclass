# Aprendizagem de Máquina - IC/UFAL

> **Minhas soluções** (Antonio Guilherme da Silva). Este repositório reúne as atividades práticas da disciplina de Aprendizagem de Máquina do IC/UFAL. Os enunciados são do Prof. Aydano Machado; as implementações nas pastas abaixo são minhas.

| Atividade | O que implementei | Técnicas |
|---|---|---|
| [01 - Pré-processamento](01_Preprocessing) | Limpeza e tratamento da base Pima Indians Diabetes (valores ausentes e implausíveis, normalização) para melhorar a acurácia de um k-NN avaliado em servidor remoto | pandas, k-NN, imputação, normalização |
| [02 - Otimização](02_Optimization) | Agentes que buscam a melhor configuração de uma antena para maximizar o ganho do padrão de radiação | Hill Climbing, Simulated Annealing, cache de avaliações |
| [03 - Validação](03_Validation) | Classificador do tipo de abalone com novas features (`meat_yield`, `shell_ratio`, `bmi`, razão comprimento/diâmetro) e busca de hiperparâmetros | Random Forest, Pipeline, `RandomizedSearchCV` |
| [06 - Clustering](06_CLUSTERING) | Segmentação de dados oftalmológicos (biometria ocular Barrett II e topografia corneana RTVue), com escolha do número de clusters e visualização | K-Means, DBSCAN, clustering hierárquico, PCA, método do cotovelo |

**Stack:** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

---

## Sobre a disciplina

Repositório da disciplina de Aprendizagem de Máquina dos cursos de Ciência da Computação e Engenharia da Computação do Instituto de Computação da Universidade Federal de Alagoas (IC/UFAL).

A disciplina tem como objetivo apresentar os principais paradigmas de aprendizagem de máquina, incluindo uma variedade de algoritmos e técnicas. Ela também contempla uma gama de atividades práticas onde os principais algoritmos serão implementados e testados em dados sintéticos e reais. Fornecendo uma visão mais próxima da resolução de problemas e aplicação da aprendizagem de máquina.

Se você está cursando a disciplina, nós encorajamos todas as atividades que fortaleçam o seu aprendizado. Tais como, formar grupos de estudo e discutir os assuntos ministrados e exercícios/atividades etc. No entanto, as respostas e soluções que você envia devem ser o seu próprio trabalho. Para os exercícios de programação, você pode discutir com outros alunos, discutir algoritmos específicos, propriedades de algoritmos, etc., pedimos apenas que você não olhe para nenhum código-fonte escrito por outra pessoa, nem passe sua solução para outro aluno.

Para tudo que nos é enviado, assumimos que você está seguindo o código de honra a seguir.

## Código de Honra

>"Como membro da comunidade deste curso, não vou participar nem tolerar a desonestidade acadêmica".

## Licença

Todas as soluções desenvolvidas por mim para este repositório estão sob a licença MIT. Vide arquivo LICENSE para mais detalhes.

## Copyright

Copyright (c) 2018 [Aydano Machado](http://www.aydanomachado.com).
