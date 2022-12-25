#!python3
#coding:utf8

#######################################
###########  Ghiles Larbi  ############
###########    Section 2   ############
###########    Groupe 5    ############
#######################################

USE_TABULATE_MODULE = True
USE_UTF8_ENCODE = True
EMPTY_CELL = "*"

#################### UTILS #######################
def get_super(x):
    return str(x)

if USE_UTF8_ENCODE :
    try:
        from sys import stdout
        '₀₁₂₃₄₅₆₇₈₉'.encode(stdout.encoding)
        def get_super(x):
            normal = "0123456789"
            super_s = "₀₁₂₃₄₅₆₇₈₉"
            res = str(x).maketrans(''.join(normal), ''.join(super_s))
            return str(x).translate(res)

    except UnicodeEncodeError:
        pass

def find(vect, maxi=True, positif=True):
    new_vect = [i for i in vect if i != None and i < 0]
    if positif:
        new_vect = [i for i in vect if i != None and i > 0]
    if new_vect == []:
        return None
    if maxi:
        return vect.index(max(new_vect))
    return vect.index(min(new_vect))

def find_cp(vect):
    if type == "MAX":
        return find(vect, maxi=True, positif=True)
    if degenere:
        return find(vect, maxi=True, positif=False)
    return find(vect, maxi=False, positif=False)

def clone_matrix(matrix):
    return [[i for i in vect] for vect in matrix]

def elemenate_fraction(nbr):
    try :
        return round(nbr) if round(nbr) == nbr else round(nbr, 2)
    except :
        return nbr

def format_equation(vect, op):
    equa = ""
    is_first = True
    for i in range(len(vect)):
        if vect[i] != 0:
            sign = " - " if vect[i] < 0 else " + "
            coeff = "" if vect[i] in [-1, 1] else str(abs(elemenate_fraction(vect[i]))) + "*"
            sign = (sign.strip() if sign == " - " else "") if is_first else sign
            equa += "{}{}x{}".format(sign, coeff, get_super(i + 1))
            is_first = False
    
    return ('0' if equa == '' else equa) + (" ≥" if op > 0 else " ≤" if op < 0 else " =")

def print_PL():
    print("\n"+"-"*40)
    print("{}   {} z".format(type, format_equation(obj_func_coeff, 0)))
    for i in range(m):
        print("{}   {} {}".format("s.c" if i == 0 else "   ",
        format_equation(constraints_coeff[i], constraints_opr[i]),
        elemenate_fraction(constraints_sm[i])))
    print("      ", end="")
    for i in range(n):
        print("x{}".format(get_super(i+1)), end=" ≥ 0\n" if i == n - 1 else ", ")
    print("-"*40, "\n")

def format_matrix():
    ADD_SM_CP_COLOMN = True
    COL = col
    new_mat = clone_matrix(simplex_matrix)
    
    if big_M :
        for i in range(COL) :
            nbr = elemenate_fraction(new_mat[-1][i])
            if nbr == None :
                continue
            Mnbr = elemenate_fraction(simplex_Mobj_func[i])
            sign = '' if Mnbr > 0 and nbr == 0 else '-' if Mnbr < 0 else '+'
            Mnbr = '' if Mnbr == 0 else sign + (str(abs(Mnbr)) if Mnbr not in [1, -1] else '') + 'M'
            new_mat[-1][i] = (str(nbr) if nbr != 0 else '' if Mnbr != '' else '0') + Mnbr

    if ADD_SM_CP_COLOMN :
        for i in range(row - 1) :
            new_mat[i] += [elemenate_fraction(constraints_sm_cp[i])]
        new_mat[-1].append(EMPTY_CELL)
        COL += 1

    # Neglect fractions
    for i in range(row):
        for j in range(COL):
            new_mat[i][j] = str(elemenate_fraction(new_mat[i][j])) if new_mat[i][j] != None else EMPTY_CELL

    # Add headers
    header = ["x" + get_super(i+1) for i in range(n)] + ["e" + get_super(i+1) for i in range(ecart_nbr)]
    header += ["t" + get_super(i+1) for i in range(art_nbr)]
    header += ["SM", "SM/CP"] if ADD_SM_CP_COLOMN else ["SM"]
    new_mat.insert(0, header)

    # Add VBS
    for i in range(row - 1) :
        new_mat[i + 1].insert(0, header[vbs[i]])
    new_mat[0].insert(0, "Base")
    new_mat[-1].insert(0, "-z")
    return new_mat

def print_matrix():
    COL = col + 2
    ROW = row + 1
    matrix = format_matrix()
    col_max_len = [0]*(COL)

    for i in range(COL) :
        col_max_len[i] = max(len(row[i]) for row in matrix)

    print()
    for i in range(ROW) :
        print("|", end='')
        for j in range(COL) :
            elem = matrix[i][j]
            print('', elem, ' '*(col_max_len[j] - len(elem)) + '|', end='')
        print()

if USE_TABULATE_MODULE :
    try :
        from tabulate import tabulate
        def print_matrix():
            new_mat = format_matrix()
            print(tabulate(new_mat[1:], tablefmt="pretty", headers=new_mat[0]))
        
    except ModuleNotFoundError:
        print("\n\n+ (!) Le module 'tabulate' n'est pas installé !!")
        print("+ (!) Cela aide à dessiner correctement les tableaux dans le terminal.")
        print("+ (!) Quittez le programme et tapez 'pip3 install tabulate' sur le terminal pour l'installer.")
        input("+ (!) Cliquez sur ENTRÉE pour continuer avec le format de tableau prédéfini : ")

#################### INPUT #######################
print("\n+ Choisir le type du PL (MAX/MIN)")
inp_err = False
while True :
    type = input("|- "+ ("(!) " if inp_err else '') + "Type du PL : ").upper()
    try : 
        if type not in ["MAX", "MIN"] :
            raise Exception()
        break
    except :
        inp_err = True

print("\n+ Saisir les coefficients de la fonction objectif séparés par un espace")
inp_err = False
while True :
    inp = input("|- "+ ("(!) " if inp_err else '') + "Fonction Objectif : ").split()
    try : 
        obj_func_coeff = [float(i) for i in inp]
        break
    except :
        inp_err = True

constraints_coeff = []
m = 0
print("\n+ Saisir les coefficients des contraintes séparés par un espace (laisser vide pour quitter)")
while True :
    inp_err = False
    while True :
        print("|- "+ ("(!) " if inp_err else '') + "Contraine", m + 1, ": ", end="")
        try : 
            inp = [float(i) for i in input().split()]
            break
        except :
            inp_err = True
    if inp == [] :
        break
    constraints_coeff += [inp]
    m += 1

constraints_sm = []
print("\n+ Saisir le Second Membre des contraintes")
for i in range(m) :
    inp_err = False
    while True : 
        print("|- "+ ("(!) " if inp_err else '') + "Contarinte", i + 1, ": ", end="")
        try :
            constraints_sm += [float(input())]
            break
        except :
            inp_err = True

constraints_opr = []
print("\n+ Saisir les operators des contraintes (1 pour '≤' | 2 pour '=' | 3 pour '≥')")
for i in range(m) :
    inp_err = False
    while True :
        print("|- "+ ("(!) " if inp_err else '') + "Contarinte", i + 1, ": ", end="")
        try :
            inp = int(input())
            if inp not in [1,2,3] :
                raise Exception()
            constraints_opr += [inp - 2]
            break
        except :
            inp_err = True


# nombre des variables de decision
n = max([len(vect) for vect in constraints_coeff + [obj_func_coeff]]) 

####################### Corriger le nombre des variables  ###########################

constraints_coeff = [constraint + [0]*(n - len(constraint)) for constraint in constraints_coeff]
obj_func_coeff += [0]*(n - len(obj_func_coeff))

##################### TEST #######################
# -------- EXO 1 Planche 3 --------
# Décommentez les lignes ci-dessous pour les tester

# type = "MAX"
# obj_func_coeff = [5,5,3]
# constraints_coeff = [[1,3,1], [-1,0,3], [2,-1,2], [2,3,-1]]
# constraints_sm = [3,2,4,2]
# constraints_opr = [-1, -1, -1, -1]
# n = 3 # nombre des variables de decision
# m = 4 # nombre des contrainte et des variable artificial

# ------- EXO 2 Planche 3 --------
# Décommentez les lignes ci-dessous pour les tester

# type = "MIN"
# obj_func_coeff = [2, -3, -4, 1]
# constraints_coeff = [[-1, -3, 1, 3], [2, 1, 1, 3], [0, -4, 2, 6]]
# constraints_sm = [2,8,4]
# constraints_opr = [-1, -1, -1]
# n = 4 # nombre des variables de decision
# m = 3 # nombre des contrainte / nombre des variable artificial

# ------- EXO 3 Planche 3 --------
# Décommentez les lignes ci-dessous pour les tester

# type = "MIN"
# obj_func_coeff = [2, 3]
# constraints_coeff = [[4, 1], [3, 2], [1, 2]]
# constraints_sm = [5, 6, 3]
# constraints_opr = [1, 1, 1]
# n = 2  # nombre des variables de decision
# m = 3  # nombre des contrainte / nombre des variable artificial
##################################################
print_PL()

####################### Corriger les inégalité ###########################
# les seconds membres doivent être positifs
is_constraints_changed = False
for i in range(m):
    if constraints_sm[i] < 0:
        is_constraints_changed = True
        constraints_sm[i] *= -1
        constraints_coeff[i] = [-i for i in constraints_coeff[i]]
        constraints_opr[i] *= -1

if is_constraints_changed :
    print("(!) Les seconds membres doivent être positifs")
    print_PL()

#####################################################################
# Vérifiez si nous pouvons résoudre le PL avec la méthode du simplex
big_M = any([False if opr == -1 else True for opr in constraints_opr])

# Calculez combien de variables artificielles et d'écart nous ajouterons
art_nbr = 0
ecart_nbr = m
for opr in constraints_opr :
    art_nbr += 1 if opr != -1 else 0
    ecart_nbr -= 1 if opr == 0 else 0

if big_M:
    print("(!) Nous allons résoudre ce PL avec la méthode de Big M\n")
else :
    print("(!) Nous allons résoudre ce PL avec la méthode de Simplex\n")
input("+ Cliquez sur ENTRÉE pour résoudre ce PL : ")

#####################################################################
# Générez la matrice des variables d'éxcedent et d'écart
# Générez la matrice des variables artificielles
ecart_coeff = [[0 for i in range(ecart_nbr)] for i in range(m)]
art_coeff = [[0 for i in range(art_nbr)] for i in range(m)]

tmp_e, tmp_a = 0, 0
for i in range(m) :
    if constraints_opr[i] != 0 :
        ecart_coeff[i][tmp_e] = - constraints_opr[i]
        tmp_e += 1
    if constraints_opr[i] != -1:
        art_coeff[i][tmp_a] = 1
        tmp_a += 1

obj_func_ecoeff = [0]*ecart_nbr
obj_func_acoeff = [0]*art_nbr

#####################################################################
# Calculer la nouvelle fonction objectif de Big M
Mobj_func_coeff  = [0]*n
Mobj_func_ecoeff = [0]*ecart_nbr
Mobj_func_acoeff  = [0]*art_nbr
Mobj_func_sm = 0

for i in range(m) :
    if constraints_opr[i] == -1:
        continue
    for j in range(n):
        Mobj_func_coeff [j] += -constraints_coeff[i][j]
    
    for j in range(ecart_nbr):
        Mobj_func_ecoeff[j] += -ecart_coeff[i][j]
    
    Mobj_func_sm += constraints_sm[i]

if type == "MAX":
    Mobj_func_coeff  = [-i for i in Mobj_func_coeff ]
    Mobj_func_ecoeff = [-i for i in Mobj_func_ecoeff]
    Mobj_func_sm *= -1

Mobj_func_sm *= -1

# déclarer un tableau des indices des variables de base pour suivre les
# changements lors de l'application de l'algorithme
vbs = [None] * m
tmp_e, tmp_t = 0, 0
for i in range(m):
    if constraints_opr[i] == -1:
        vbs[i] = n + tmp_e
        tmp_e += 1
        continue
    else :
        vbs[i] = n + ecart_nbr + tmp_t
        tmp_t += 1

##################### Générer la matrice de simplex #######################
simplex_matrix = clone_matrix(constraints_coeff)
for i in range(m) :
    simplex_matrix[i] += ecart_coeff[i] + art_coeff[i] + [constraints_sm[i]]
simplex_matrix += [[i for i in obj_func_coeff]]
simplex_matrix[-1] += obj_func_ecoeff + obj_func_acoeff +  [0]

simplex_Mobj_func = Mobj_func_coeff  + Mobj_func_ecoeff + Mobj_func_acoeff  + [Mobj_func_sm]

col = n + ecart_nbr + art_nbr + 1
row = m + 1

##################### Appliqué l'algoritheme #######################
degenere = False
k = 0

constraints_sm_cp  = [0]*m
obj_func_sm_cp = 0

while True :
    # Trouver la colonne de pivot
    if big_M :
        cp = find_cp(simplex_Mobj_func[: n + ecart_nbr])
    if not(big_M) or (big_M and (cp == None and all(not i for i in simplex_Mobj_func[: n + ecart_nbr]))):
        cp = find_cp(simplex_matrix[-1][: n + ecart_nbr])
    
    # Test de critere d'arret
    if cp == None:
        break

    # Calculer SM/CP
    for i in range(m):
        if simplex_matrix[i][cp] == 0 :
            constraints_sm_cp[i] = None
            continue
        constraints_sm_cp[i] = simplex_matrix[i][-1] / simplex_matrix[i][cp]
    
    # Trouver la ligne de pivot
    lp = find(constraints_sm_cp, maxi=False, positif=True)

    ###############
    print("\nK =", k)
    print_matrix()
    ##############

    if lp == None:
        print("(!) La solution est Infini")
        exit(1)
    
    # Trouver le pivot
    pivot = simplex_matrix[lp][cp]

    ###########################################
    print("\ncp =", cp + 1, end="  |  ")
    print("lp =", lp + 1, end="  |  ")
    print("pivot =", elemenate_fraction(pivot))
    print("-" * 34, "\n")
    input("+ Cliquez sur ENTRÉE pour continuer : ")
    ###########################################

    # Nouvelle variable de base
    vbs[lp] = cp
    k += 1

    ############ Algorithme de GAUSS ##############
    simplex_matrix[lp] = [nbr / pivot if nbr != None else None for nbr in simplex_matrix[lp]]

    # Calculer la nouvelle matrice
    old_mat = clone_matrix(simplex_matrix)
    for i in range(row):
        if i == lp:
            continue
        for j in range(col):
            try :
                simplex_matrix[i][j] = old_mat[i][j] - old_mat[i][cp] * old_mat[lp][j]
            except :
                pass

    if big_M:
        old_Mobj_func = simplex_Mobj_func.copy()
        for i in range(col):
            try :
                simplex_Mobj_func[i] = old_Mobj_func[i] - old_Mobj_func[cp] * old_mat[lp][i]
            except :
                pass

        # Éliminer le variable sortante
        for i in range(n + ecart_nbr, n + ecart_nbr + art_nbr):
            if i not in vbs:
                for j in range(row):
                    simplex_matrix[j][i] = None
                simplex_Mobj_func[i] = None
    
    # Verifier si la solution est dégénéré
    for i in range(row - 1):
        if (simplex_matrix[i][-1] == 0):
            print("\n(!) La solution est dégénéré")
            print_matrix()

            if degenere:
                print("(!) Nous ne pouvons pas appliquer la règle de Bland pour la deuxième fois ...\n")
                exit(1)

            input("\n+ Cliquez sur ENTRÉE pour appliquer la règle de Bland : ")
            degenere = True
            simplex_matrix = clone_matrix(old_mat)
            if big_M:
                simplex_Mobj_func = old_Mobj_func.copy()
            break

##################### Afficher le resultat #######################
print("\nK =", k)
print_matrix()
print("\nTous les coefficients de la fonction objectif sont {} alors le critére d'arret est verifié\n".format("négatifs" if type == "MAX" else "positifs"))

optimal_sol = [0] * n
for i in range(m):
    if vbs[i] < n:
        optimal_sol[vbs[i]] = elemenate_fraction(simplex_matrix[i][-1])

print("La solution optimale :\nx* = (", end="")
for i in range(n):
    print(optimal_sol[i], end="" if i == n - 1 else ", ")
print(")\n")

print("L'optimume :\nz* =", -elemenate_fraction(simplex_matrix[-1][-1]), "\n")