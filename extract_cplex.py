import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ExtractCPLEX:

  def findPos(self,file,spected, starter = "Node  Left"):
    file = open(file,"r")
    for i,j in enumerate(file):
      if starter in j:
        return [(j.index(k)+len(k)-1) for k in spected]

  def getFilelog(self,f):
    with open(f, 'r') as file:
      data = file.read().replace('\n', '')
    tmp = []
    for i in re.findall(r"\w*Logfile\s*\'*[a-zA-z.+]*\'",data):
      tmp.append(i.replace("'","").split(" ")[1])
    return tmp

  def createTables(self,file):
    table_start = False
    spected = ['Node', 'Left', 'Objective', 'IInf', 'Integer', 'Bound', 'ItCnt', 'Gap']
    expectedPositions = self.findPos(file,spected)
    tables = {"seconds":[],"ticks":[],
              "solution":[],"Node":[],
              "Left":[],"Objective":[],
              "IInf":[],"Integer":[],
              "Bound":[],"ItCnt":[],
              "Gap":[],"cuts":[]
              }
    time = None
    ticks = None
    cuts = None
    f = open(file,"r")
    for i in f:
      if re.findall(r"\d{1,}[+]{1}\d{1,}",i):
        i = " ".join(i.split("+"))
      if("Cover cuts applied" in i or "Performing restart 1" in i):
        table_start = False
      if("Elapsed time" in i):
          tmp = [float(k) for k in re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)]
          time = tmp[0]
          ticks = tmp[1]

      if(table_start):
        if((str(i)[0] == " " or str(i)[0] == "*") and str(i)[expectedPositions[0]].isdigit()):
          i = i.replace("uts: "," uts:")
          tables["seconds"].append(time)
          tables["ticks"].append(ticks)
          tables["solution"].append(1 if i[0]=="*" else 0)
          tables["cuts"].append(None)
          for j,m in zip(expectedPositions,spected):
            if i[j] != " ":
              tmp = ""
              for k in range(j,0,-1):
                if i[k] != " ":
                  tmp += i[k]
                else:
                  break
              tmp = tmp[::-1]
              if("ut" in tmp.lower() or "infeasible" in tmp.lower() or "integral" in tmp.lower()):
                if("uts" in tmp.lower()):
                  tables["cuts"][-1] = int(tmp.split(":")[1])
                tables[m].append(tables[m][-1])
              else:
                if m == "Gap":
                  tables[m].append(float(tmp)/100)
                else:
                  tables[m].append(float(tmp))
            else:
              tables[m].append(None)
              
      if("Node  Left" in i):
        table_start = True
    return tables

  def extract(self,file):
    variables = {"mipPresolveEliminated":[],
                 "mipPresolveModified":[],
                 "aggregatorDid":[],
                 "reducedMipHasColumns":[],
                 "reducedMipHasNonZero":[],
                 "reducedMipHasBinaries":[],
                 "reducedMipHasGeneral":[],
                 "cliqueTableMembers":[],
                 "rootRelaxSolSeconds":[],
                 "rootRelaxSolTicks":[]}
    variables["logFile"] = self.getFilelog(file)
    tables = self.createTables(file)

    f = open(file, "r")
    for i in f:    
      if("linear optimization" in i):
        variables["linearOpt"] = float(i.replace(" ","").replace("\n","").split(":")[1])
      if("optimality gap tolerance" in i):
        variables["gapTol"] = float(i.replace(" ","").replace("\n","").split(":")[1])
      if("time limit in seconds" in i):
        variables["timeLimit"] = float(i.replace(" ","").replace("\n","").split(":")[1])
      if("emphasis for MIP optimization" in i):
        variables["mipOpt"] = float(i.replace(" ","").replace("\n","").split(":")[1])
      if("Objective sense" in i):
        variables["objSense"] = i.replace(" ","").replace("\n","").split(":")[1]
      if("Variables" in i):
        if("Box:" in i):
          variablesValue = ["variablesValue","Nneg","Box","Binary"]
          for j,k in enumerate(re.findall(r'\d+', i.replace(" ","").replace("\n","").split(":",1)[1])):
            variables[variablesValue[j]] = float(k)
        else:
          variablesValue = ["minLB","maxUb"]
          for j,k in enumerate(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)):
            variables[variablesValue[j]] = float(k)
      if("Objective nonzeros" in i):
        if("Min" in i or "Max" in i):
          variablesValue = ["objNonZerosMin","objNonZerosMax"]
          for j,k in enumerate(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)):
            variables[variablesValue[j]] = float(k)
        else:
          variables["objNonZeros"] = float(i.replace(" ","").replace("\n","").split(":")[1])
      if("Linear constraints" in i):
        if("Less" in i):
          variablesValue = ["linearConstraintsValue","less","greater","equal"]
          for j,k in enumerate(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)):
            variables[variablesValue[j]] = float(k)
        else:
          pass
      if("Nonzeros" in i):
        if("Min" in i):
          variablesValue = ["nonZerosMin","nonZerosMax"]
          for j,k in enumerate(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)):
            variables[variablesValue[j]] = float(k)
        else:
          variables["nonZeros"] = float(i.replace(" ","").replace("\n","").split(":")[1])
      if("RHS nonzeros" in i):
        if("Min" in i):
          variablesValue = ["rhsNonZerosMin","rhsNonZerosMax"]
          for j,k in enumerate(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)):
            variables[variablesValue[j]] = float(k)
        else:
          variables["rhsNonZeros"] = float(i.replace(" ","").replace("\n","").split(":")[1])
      if("CPXPARAM_TimeLimit" in i):
        variables["CPXPARAM_TimeLimit"] = float(i.replace("\n","").split(" ")[-1])
      if("MIP Presolve eliminated" in i):
        variables["mipPresolveEliminated"].append([int(k) for k in re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)])
      if("MIP Presolve modified " in i):
        variables["mipPresolveModified"].append(int(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)[0]))
      if("Reduced MIP has" in i):
        if("indicators." in i):
          tmp = [int(k) for k in re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)]
          variables["reducedMipHasBinaries"].append(tmp[0])
          variables["reducedMipHasGeneral"].append(tmp[1])
        else:
          tmp = [int(k) for k in re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)]
          variables["reducedMipHasColumns"].append(tmp[1])
          variables["reducedMipHasNonZero"].append(tmp[-1])
          reduceHasGeneral = []
      if("Clique" in i):
        variables["cliqueTableMembers"].append(float(i.replace(" ","").replace("\n","").split(":")[1]))
      if("Aggregator did" in i):
        variables["aggregatorDid"].append(int(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)[0]))
      if("Root relaxation" in i):
        tmp = [float(k) for k in re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?',i)]
        variables["rootRelaxSolSeconds"].append(tmp[0])
        variables["rootRelaxSolTicks"].append(tmp[1])
      if("Lift and" in i):
        variables["liftAndProjectCuts"] = int(i.replace(" ","").replace("\n","").split(":")[1])
      if("Gomory fractional" in i):
        variables["gomoryFract"] = int(i.replace(" ","").replace("\n","").split(":")[1])
    df=pd.DataFrame.from_dict(tables).rename(columns={
      "seconds":"seconds",
      "ticks":"ticks",
      "solution":"solution",
      "Node":"node",
      "Left":"nodesLeft",
      "Objective":"objective",
      "IInf":"iinf",
      "Integer":"bestInteger",
      "Bound":"BestBound",
      "ItCnt":"itCnt",
      "Gap":"gap",
      "cuts":"cuts"})
    ## Ajuste de lectura de datos del log CPLEX
    # df = df[ df['bestInteger'] > 1 ]

    df['seconds'] = df['seconds'].fillna(0)
    eps=np.arange(0, 1, 1/(len(df)), dtype=float)
    df['eps'] = eps
    sum_column = df["seconds"] + df["eps"]
    df["seconds"] = sum_column
    df['ticks'] = df['ticks'].fillna(0)
    eps=np.arange(0, 1, 1/(len(df)), dtype=float)
    df['eps'] = eps
    sum_column = df["ticks"] + df["eps"]
    df["ticks"] = sum_column
    df = df[ df['seconds'] > 10 ]
    return df , variables
  
  def mi_funcion_extract(self, file):
    e = ExtractCPLEX()
    df, dic = e.extract(file)
    return df, dic

  def filtrar_valores_atipicos(self,df):
      # Calcular el rango intercuartílico para "bestInteger" y "BestBound"
      Q1 = df[['bestInteger', 'BestBound']].quantile(0.25)
      Q3 = df[['bestInteger', 'BestBound']].quantile(0.75)
      IQR = Q3 - Q1
      # Definir el rango para filtrar los valores atípicos
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      # Aplicar el filtro y devolver el DataFrame filtrado
      df_filtrado = df[
          (df['bestInteger'] >= lower_bound['bestInteger']) & (df['bestInteger'] <= upper_bound['bestInteger']) &
          (df['BestBound'] >= lower_bound['BestBound']) & (df['BestBound'] <= upper_bound['BestBound'])
      ]
      return df_filtrado

  def calcular_y_guardar_gap(self,df):    
      # Calcular la columna 'gap' utilizando la función gap
      df['gap'] = df.apply(lambda row: gap(row['BestBound'], row['bestInteger']), axis=1)

      # Redondear la columna 'seconds' a dos cifras decimales
      df['seconds'] = df['seconds'].round(1)
      df['ticks'] = df['ticks'].round(1)
      # Convertir las columnas 'node' y 'nodesLeft' a enteros
      df['node'] = df['node'].astype(int)
      df['nodesLeft'] = df['nodesLeft'].astype(int)
      # Eliminar las columnas 'cuts' y 'eps'
      df = df.drop(['cuts', 'eps','iinf','itCnt'], axis=1)
      ## Elimina filas con algún campo vacio
      df = df.dropna()
      df= df[df['gap'] <= df['gap'].shift(1)]
      # Guardar el DataFrame con la nueva columna 'gap' en un archivo CSV
      df.to_csv("convergencia.csv", index=False)
      return df