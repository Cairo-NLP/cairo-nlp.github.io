+++
title = "1. Abstraction of NLP - تجريدُ معالجةِ اللغاتِ الطبيعية"
date = "2024-09-11"
draft = false
written_by = "Mayar Osama"
translated_by = "AbdulRahman Ateyya"

+++


We as humans, by nature, understand text "natural language"; machines, by nature, understand numbers (in one way or another numerical representation).  
If we want to visualize any NLP task, it could be viewed as:

1. Taking text as input.
2. Preprocessing it and getting the numerical representation of this text.
3. Passing it to a model to "understand" the input to extract the needed features and perform the given task.
4. Taking the output from the model, which would be represented as some numerical representation as well, and post-processing it to convert this output into some text (if needed).

---

<div class="arabic-content">

تجريدُ معالجةِ اللغاتِ الطبيعية: 
نحن -البشرَ- بطبيعتِنا نفهمُ النصوصَ (اللغةَ الطبيعيةَ)، أما الآلاتُ فطبيعتُها أنها تفهمُ الأرقامَ (تمثيلاتٍ رقميةً بصورةٍ أو بأخرى).
إذا أرَدْنا أن نستعرضَ إحدى مهماتِ معالجةِ اللغاتِ، فسنجدُها كالآتي:
1.	تحصيلُ النصِّ المُدْخََل.
2.	تحضيرُه للمعالجة (وهو ما يسمى بالمعالجةِ المسبقة).
3.	تحويلُ النصِّ إلى صيغةٍ رقميةٍ.
4.	إدخالُ الصيغةِ الرقميةِ إلى النموذجِ لفهمِ النصِّ.
5.	استخلاصُ السماتِ اللازمةِ للمعالجةِ.
6.	إجراءُ المهمةِ/العمليةِ المطلوبةِ.
7.	الآنَ حصلنا على مُخْرَجَاتِ النموذج.
8.	أخيرًا، معالجةُ هذه المُخْرَجاتِ لتحويلِها إلى صورةٍ مفهومةٍ (نصيّةٍ) بحسبِ ما تقتضيه المهمة.

</div>