# salvar como make_latex_rows.py
import pandas as pd

# ajuste de mapeamentos legíveis
MAP_OBST = {"none": "Sem", "many_small": "Muitos Peq.", "few_large": "Poucos Gran."}

def extrair_campos(id_str):
    partes = id_str.split('_')
    num = partes[1]
    # obstáculo geralmente em partes[3] no seu padrão
    obst = partes[3] if len(partes) > 3 else ""
    borda = "Com" if "with_border" in id_str else "Sem"
    spray = next((p.replace("spray", "") for p in partes if "spray" in p), "")
    return num, obst, borda, spray

def format_num(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "0.00"

def main():
    df = pd.read_csv("resultados_formatados.csv")
    # garantir nomes limpos
    df.columns = df.columns.str.strip()
    # extrair campos do ID
    df[['Num','_obst_raw','Borda','Spray']] = df['ID'].apply(
        lambda s: pd.Series(extrair_campos(s))
    )
    # mapear obstáculo para forma amigável
    df['Obstáculo'] = df['_obst_raw'].map(MAP_OBST).fillna(df['_obst_raw'])
    # deixamos a coluna Estratégia (já existe no CSV)
    # ordenar por Num (numérico) e estratégia (mantém ordem consistente)
    df['Num_int'] = df['Num'].astype(int)
    df = df.sort_values(by=['Num_int', 'Spray', 'Estratégia'])

    # agrupar por Num e imprimir o bloco LaTeX
    for num, group in df.groupby('Num', sort=True):
        # unifica linhas por instância
        rows = list(group.to_dict('records'))
        print(r"\addlinespace")
        # if there are both strategies, try to print best first then first
        # otherwise print whatever rows exist (in the group order)
        # build helper to find by strategy
        def find_strategy(lst, strat):
            for r in lst:
                if str(r.get('Estratégia','')).lower() == strat:
                    return r
            return None

        best = find_strategy(rows, 'best')
        first = find_strategy(rows, 'first')

        # helper to print a full line (prefix) or the follow-up line (blanks)
        def print_full(r):
            num_s = f"{int(r['Num']):02d}"
            spray = r.get('Spray','')
            obst = r.get('Obstáculo','')
            borda = r.get('Borda','')
            strat = r.get('Estratégia','')
            obj_ini = format_num(r.get('Obj Inicial', 0))
            obj_fin = format_num(r.get('Obj Final', 0))
            mel = format_num(r.get('Melhoria (%)', 0))
            cob = format_num(r.get('Cobertura Final (%)', 0))
            tempo = format_num(r.get('Tempo (s)', 0))
            print(f"{num_s} & {spray} & {obst} & {borda} & {strat} & {obj_ini} & {obj_fin} & {mel} & {cob} & {tempo} \\\\")

        def print_followup(r):
            strat = r.get('Estratégia','')
            obj_ini = format_num(r.get('Obj Inicial', 0))
            obj_fin = format_num(r.get('Obj Final', 0))
            mel = format_num(r.get('Melhoria (%)', 0))
            cob = format_num(r.get('Cobertura Final (%)', 0))
            tempo = format_num(r.get('Tempo (s)', 0))
            # blank prefix exactly like your example: "   &    &             &     & first & ..."
            print(f"   &    &             &     & {strat} & {obj_ini} & {obj_fin} & {mel} & {cob} & {tempo} \\\\")

        if best and first:
            print_full(best)
            print_followup(first)
        else:
            # fallback: print in the group order, first row full, then any others as followup
            if len(rows) >= 1:
                print_full(rows[0])
            for r in rows[1:]:
                print_followup(r)

if __name__ == "__main__":
    main()
