# Otimização de Rotas de Pulverização Agrícola via Busca Tabu

Este repositório contém o código-fonte do projeto desenvolvido para a disciplina MO824/MC859 (Tópicos em Otimização Combinatória) da Unicamp.

O projeto implementa uma meta-heurística de **Busca Tabu** para resolver um problema de otimização de rotas de pulverização agrícola, modelado como uma variação do *Covering Salesman Problem* (CSP). O objetivo é encontrar uma sequência de pontos (rota) que minimize um custo operacional total, composto pela distância percorrida, penalidade por falha de cobertura e penalidade por complexidade de manobra.

## Organização do Código-Fonte

O código-fonte do projeto foi estruturado de forma modular para facilitar o desenvolvimento, a manutenção e a experimentação. A estrutura de diretórios principal e os componentes mais importantes são descritos a seguir.

### Estrutura de Diretórios

A raiz do projeto contém os scripts principais de execução e os diretórios dos pacotes.

* `run_experiments.py`: Script principal para executar os experimentos de forma automatizada, iterando sobre as instâncias e configurando os parâmetros do solver.
* `instance_generation.ipynb`: Notebook Jupyter utilizado para gerar, visualizar e salvar o conjunto de instâncias de teste.
* `agricultural_csp/`: Pacote Python que encapsula toda a lógica central do problema.
* `tests/`: Contém notebooks para testes, validação de componentes e análises exploratórias.

### Pacote `agricultural_csp`

Este pacote é o núcleo do projeto e contém as seguintes classes e módulos principais:

* **`instance.py`**: Define a classe `AgcspInstance`, que armazena todos os dados estáticos de uma instância do problema, como o grid, a localização dos alvos, os obstáculos, o comprimento do pulverizador e o ângulo máximo de curva.
* **`solution.py`**: Define a classe `AgcspSolution`, que representa uma solução para o problema. Essencialmente, ela encapsula o caminho (uma lista de nós) percorrido pelo veículo.
* **`evaluator.py`**: Contém a classe `AgcspEvaluator`, um dos componentes mais críticos do projeto. Esta classe é responsável por:
    * Calcular a função objetivo e seus três componentes: proporção de cobertura, distância percorrida e penalidade por manobras.
    * Implementar métodos de **avaliação delta** (*delta evaluation*) para os movimentos de vizinhança (inserção, remoção, troca e movimento). Esses métodos calculam a variação no custo da solução de forma incremental e eficiente, sem a necessidade de reavaliar a solução completa, o que é fundamental para a performance de meta-heurísticas como a Busca Tabu.
    * Validar a factibilidade de uma solução, verificando colisões com obstáculos e violações de restrições de manobra.
* **`solver/`**: Este subdiretório contém as implementações dos algoritmos de resolução.
    * `constructive_heuristics/`: Módulos com heurísticas construtivas para gerar soluções iniciais.
    * `agcsp_ts.py`: Implementação principal do algoritmo de **Busca Tabu (Tabu Search)**. Ele utiliza o `AgcspEvaluator` para explorar o espaço de busca e encontrar soluções de alta qualidade.
* **`tools.py`**: Módulo com funções utilitárias, como a função para visualização gráfica das instâncias e soluções.

### Fluxo de Execução

O fluxo de interação entre os componentes ocorre da seguinte maneira:

1.  Uma `AgcspInstance` é carregada a partir de um arquivo.
2.  Um `AgcspEvaluator` é inicializado com essa instância.
3.  Uma heurística construtiva gera uma `AgcspSolution` inicial.
4.  O solver de Busca Tabu (`agcsp_ts.py`) recebe a solução inicial e o avaliador. A partir daí, ele melhora iterativamente a solução, utilizando os métodos de avaliação delta do avaliador para guiar a busca de forma eficiente.
