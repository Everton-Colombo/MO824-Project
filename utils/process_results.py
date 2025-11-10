import pandas as pd
import csv

def processar_resultados(caminho_entrada, caminho_saida, log_bad_lines=None):
    """
    Formata o CSV de resultados, selecionando apenas as colunas relevantes,
    renomeando, convertendo valores numéricos e arredondando.
    """
    # Lê o CSV, pulando linhas ruins
    if log_bad_lines:
        bad_lines = []
        def bad_line_handler(line):
            bad_lines.append(line)
            return None
        df = pd.read_csv(caminho_entrada, engine='python', on_bad_lines=bad_line_handler)

    else:
        df = pd.read_csv(caminho_entrada, engine='python', on_bad_lines='skip')

    # Seleciona e renomeia as colunas que realmente usamos
    df = df[[
        "Run Name", "Solver", "Strategy", "Exec Time (s)", 
        "Initial Obj", "Final Obj", "Improvement (%)", "Final Coverage (%)"
    ]].rename(columns={
        "Run Name": "ID",
        "Solver": "Solver",
        "Strategy": "Estratégia",
        "Exec Time (s)": "Tempo (s)",
        "Initial Obj": "Obj Inicial",
        "Final Obj": "Obj Final",
        "Improvement (%)": "Melhoria (%)",
        "Final Coverage (%)": "Cobertura Final (%)"
    })

    # Converte colunas numéricas
    colunas_numericas = ["Tempo (s)", "Obj Inicial", "Obj Final", "Melhoria (%)", "Cobertura Final (%)"]
    for col in colunas_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

    # Exporta CSV formatado
    df.to_csv(
        caminho_saida,
        index=False,
        sep=',',
        quoting=csv.QUOTE_ALL
    )

    print(f"Arquivo '{caminho_saida}' gerado com sucesso.")

    # Salva log de linhas ruins, se necessário
    if log_bad_lines and bad_lines:
        with open(log_bad_lines, "w", encoding="utf-8") as f:
            for line in bad_lines:
                f.write(",".join(line) + "\n")
        print(f"Linhas puladas foram registradas em '{log_bad_lines}'.")


if __name__ == "__main__":
    processar_resultados(
        "ttt_run_results.csv", 
        "ttt_resultados_formatados.csv",
        log_bad_lines="linhas_puladas.log"
    )
