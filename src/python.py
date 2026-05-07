# Імпорт бібліотек
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ================================
# 1. Завантаження даних
# ================================
df = pd.read_csv("StudentsPerformance.csv")

print("Перші 5 рядків:")
print(df.head())

# ================================
# 2. Очищення даних
# ================================
print("\nПропущені значення:")
print(df.isnull().sum())

# Видалення дублікатів
df.drop_duplicates(inplace=True)

# Перевірка типів
print("\nТипи даних:")
print(df.dtypes)

# ================================
# 3. Дескриптивна статистика
# ================================
math_scores = df['math score']

print("\n--- Дескриптивна статистика ---")
print("Середнє:", math_scores.mean())
print("Медіана:", math_scores.median())
print("Мода:", math_scores.mode()[0])

print("Дисперсія:", math_scores.var())
print("Стандартне відхилення:", math_scores.std())

print("Асиметрія:", math_scores.skew())
print("Ексцес:", math_scores.kurt())

# ================================
# 4. Візуалізація
# ================================

# Гістограма
plt.figure()
plt.hist(df['math score'], bins=20)
plt.title("Розподіл оцінок з математики")
plt.xlabel("Оцінка")
plt.ylabel("Кількість")
plt.show()

# Boxplot
plt.figure()
df.boxplot(column='math score', by='gender')
plt.title("Оцінки з математики за статтю")
plt.suptitle("")
plt.show()

# ================================
# 5. Перевірка гіпотези (T-test)
# ================================
male = df[df['gender'] == 'male']['math score']
female = df[df['gender'] == 'female']['math score']

t_stat, p_val = stats.ttest_ind(male, female)

print("\n--- T-test ---")
print("t-статистика:", t_stat)
print("p-value:", p_val)

if p_val < 0.05:
    print("Відхиляємо H0 — різниця статистично значуща")
else:
    print("Не відхиляємо H0")

# ================================
# 6. Альтернативний аналіз
# ================================
group1 = df[df['test preparation course'] == 'completed']['math score']
group2 = df[df['test preparation course'] == 'none']['math score']

t_stat2, p_val2 = stats.ttest_ind(group1, group2)

print("\n--- Альтернативний T-test ---")
print("t-статистика:", t_stat2)
print("p-value:", p_val2)

if p_val2 < 0.05:
    print("Підготовчі курси впливають на результати")
else:
    print("Вплив курсів не доведено")

# ================================
# 7. Кореляційний аналіз
# ================================
corr, p_corr = stats.pearsonr(df['math score'], df['reading score'])

print("\n--- Кореляція ---")
print("Коефіцієнт кореляції:", corr)
print("p-value:", p_corr)

if p_corr < 0.05:
    print("Є статистично значущий зв'язок між показниками")
else:
    print("Зв'язок не доведений")