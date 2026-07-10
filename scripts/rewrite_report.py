"""Rewrite report body (sections 1-end) from current notebook outputs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "4022262035-4021262131-AmirHoseinAbolfazli-ArmanBijari.tex"

phase3_rows = (ROOT / "report" / "phase3_table_rows.tex").read_text(encoding="utf-8")
vec_avg_rows = (ROOT / "report" / "vec_avg_rows.tex").read_text(encoding="utf-8")
peak_rows = (ROOT / "report" / "peak_f1_rows.tex").read_text(encoding="utf-8")

body = r"""
\section{معرفی داده‌ها}

داده‌ای که توی این پروژه باهاش کار کردیم \lr{Snapfood Persian Sentiment Analysis} هستش که از \lr{Kaggle}\footnote{\lr{Kaggle}: پلتفرم آنلاین برای مسابقات یادگیری ماشین و به اشتراک‌گذاری دیتاست.} برداشتیم. داده \lr{70,000} تا نظر کاربرای \lr{Snapfood} داره که بعد از سفارش غذا نوشتن. هر نظر یه برچسب داره: \lr{HAPPY} (مثبت) یا \lr{SAD} (منفی). بعد از فیلتر اولیه \lr{69,480} ردیف مونده (\lr{HAPPY}: \lr{34,916} / \lr{SAD}: \lr{34,564}) و بعد از پاکسازی \lr{69,479} ردیف (\lr{1} نظر خالی حذف شد). از نظر تعداد کلاس متوازنه.

یه چیزی که داده رو جالب و در عین حال چالش‌برانگیز می‌کنه اینه که نظرات کاملاً واقعی و عامیانه‌ان. غلط نگارشی، کشیدن حروف، \lr{emoji}، قاطی فارسی-انگلیسی، همه هست. اینا نویز حساب میشن ولی بخشی‌شون خودش سیگنال احساسه.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{../outputs/figures/eda_distribution.png}
    \caption{توزیع برچسب‌های مثبت و منفی در داده}
    \label{fig:eda_dist}
\end{figure}

پس داده خوبیه، ولی چون عامیانه و غیررسمیه، پیش‌پردازش درست خیلی مهمه.

یه چیز دیگه که توی ارزیابی مدل خیلی اثر میذاره اینه که برچسب‌ها همیشه تمیز نیستن. حدود \lr{22\%} نظرها «ولی» یا «اما» دارن؛ حدود \lr{6\%} همزمان کلمات مثبت و منفی دارن. برای انسان قابل فهمه، ولی برای مدلی که فقط یه برچسب \lr{HAPPY}/\lr{SAD} می‌بینه گیج‌کننده‌ست. پس سقف دقت فقط با مدل بهتر بالا نمیره، بخشی‌ش سقف ذاتی برچسب‌گذاریه.


\section{فاز اول: پیش‌پردازش متن}

\subsection{چرا فارسی سخت‌تره؟}

یه مشکل خاص فارسی اینه که بعضی حروف با عربی یونیکد مشترک ندارن ولی عین هم به نظر میان. «ی» فارسی \lr{U+06CC} و «ي» عربی \lr{U+064A}. از دید مدل دو کلمه جدا میشن مگر اینکه اول نرمال‌سازی کنیم.

\subsection{مراحل پیش‌پردازش}

کد توی \lr{\texttt{src/preprocessing.py}} هست. برخلاف یه pipeline عمومی، اینجا عمداً سیگنال احساس رو نگه می‌داریم:

اول \lr{URL} حذف میشه (قبل از \lr{Hazm}\footnote{\lr{Hazm}: کتابخونه پایتون برای پردازش زبان طبیعی فارسی.} چون نرمالایزر \lr{://} رو می‌بلعه). بعد \lr{emoji}های احساسی به \lr{EMO\_POS}/\lr{EMO\_NEG} نگاشت میشن، نه اینکه کورکورانه حذف بشن. نرمال‌سازی فارسی، حذف اعداد و علائم، جمع‌کردن حروف تکراری (خوووب $\rightarrow$ خوب)، توکن‌سازی، و مهم‌تر از همه \textbf{برچسب‌گذاری نفی}: بعد نفی‌کننده‌ها مثل «نه» به کلمات بعدی پیشوند \lr{NEG\_} می‌خوره (ایده \lr{Pang \& Lee}). حذف \lr{stopword} پیش‌فرض \textbf{خاموش}ه چون لیست \lr{Hazm} نفی‌کننده و شدت‌دهنده داره. آخر سر \lr{lemmatization} با \lr{Hazm.Lemmatizer()}؛ توکن‌های \lr{EMO\_}/\lr{NEG\_} دست نمی‌خورن.

\subsection{نمونه‌هایی از قبل و بعد}

\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{X X}
\toprule
\textbf{متن اصلی} & \textbf{بعد از پیش‌پردازش} \\
\midrule
سلام خیلی خوووووب بود! غذا عالیییی بود & سلام خیلی خوب بود غذا عالی بود \\
\addlinespace
بدترین تجربه زندگیم بود. اصلا توصیه نمیکنم & بدترین تجربه زندگیم بود اصلا توصیه کرد به هیچ‌کس \\
\addlinespace
غذا وقتی رسید سرد بوووووود. افتضاح & غذا وقتی رسید سرد بوود افتضاح \\
\bottomrule
\end{tabularx}
\caption{نمونه‌های واقعی از نوت‌بوک فاز اول}
\label{tab:preprocessing_examples}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.65\textwidth]{../outputs/figures/token_distribution.png}
    \caption{توزیع تعداد توکن (میانگین \lr{17}، میانه \lr{13})}
    \label{fig:token_dist}
\end{figure}

\subsection{تقسیم داده}

با \lr{stratified split}\footnote{\lr{stratified split}: نسبت کلاس‌ها توی train و test یکی بمونه.} و \lr{random\_state=42}: آموزش \lr{55,583} (\lr{HAPPY} \lr{27,933} / \lr{SAD} \lr{27,650})، آزمون \lr{13,896} (\lr{HAPPY} \lr{6,983} / \lr{SAD} \lr{6,913}). میانگین طول متن \lr{89.7} کاراکتر (میانه \lr{66}).


\section{فاز دوم: \lr{Vectorization}}

سه روش اجباری پروژه رو پیاده کردیم، به علاوه دو نمایش bonus که توی فازهای بعدی مقایسه شدن.

\subsection{\lr{CountVectorizer} و \lr{TF-IDF}}

هر دو sparse با \lr{max\_features=50,000}، \lr{ngram\_range=(1,2)}، \lr{min\_df=2}. \lr{TF-IDF} با \lr{sublinear\_tf=True}. هر دو ماتریس \lr{50,000} بعدی با پراکندگی \lr{99.9466\%} ساختن.

\subsection{\lr{Word2Vec}}

\lr{Skip-gram}، \lr{vector\_size=100}، آموزش از صفر روی کورپوس. میانگین بردار کلمات برای هر سند. \lr{OOV} روی train حدود \lr{53.9\%} — برای نظرات کوتاه و عامیانه طبیعیه.

\subsection{نمایش‌های bonus}

\lr{Word2Vec-IDF}: میانگین وزن‌دار با \lr{IDF} (ایده شبیه \lr{SIF}). \lr{Hybrid word+char TF-IDF}: اتحاد \lr{word} و \lr{char\_wb} n-gram؛ \lr{101,119} بعد با پراکندگی \lr{99.7875\%}. برای املای غلط و کشیدگی فارسی خیلی کمک می‌کنه.

\begin{table}[H]
\centering
\small
\begin{tabular}{lccc}
\toprule
\textbf{نمایش} & \textbf{بعد} & \textbf{نوع} & \textbf{پراکندگی} \\
\midrule
\lr{CountVectorizer} & \lr{50,000} & sparse & \lr{99.9466\%} \\
\lr{TF-IDF} & \lr{50,000} & sparse & \lr{99.9466\%} \\
\lr{Word2Vec (mean)} & \lr{100} & dense & --- \\
\lr{Word2Vec-IDF} & \lr{100} & dense & --- \\
\lr{Hybrid} & \lr{101,119} & sparse & \lr{99.7875\%} \\
\bottomrule
\end{tabular}
\caption{خلاصه نمایش‌های عددی}
\label{tab:vec_summary}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{../outputs/figures/vectorizer_norms.png}
    \caption{توزیع نُرم بردارها}
    \label{fig:vec_norms}
\end{figure}


\section{فاز سوم: آموزش و ارزیابی مدل‌های پایه}

\subsection{مدل‌ها و تنظیمات}

نه مدل کلاسیک روی پنج \lr{vectorizer} (سه اجباری + دو bonus): \lr{ComplementNB}، \lr{GaussianNB}، \lr{SVM (Linear)} داخل \lr{CalibratedClassifierCV}، \lr{LogisticRegression}، \lr{DecisionTree}، \lr{RandomForest}، \lr{KNN}، \lr{AdaBoost}، \lr{HistGradientBoosting}. \lr{ComplementNB} فقط روی sparse غیرمنفی؛ \lr{GaussianNB} روی \lr{Word2Vec}؛ \lr{HistGradBoost} روی dense. جمعاً \lr{32} آزمایش در sweep اولیه، بعد \lr{GridSearchCV} برای سه مدل خطی/NB برتر.

\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{l X}
\toprule
\textbf{مدل} & \textbf{تنظیمات sweep} \\
\midrule
\lr{ComplementNB / GaussianNB} & \lr{alpha=0.1} \\
\lr{SVM (Linear)} & \lr{C=1.0, max\_iter=4000} در \lr{CalibratedClassifierCV} \\
\lr{LogisticRegression} & \lr{C=1.0, solver=lbfgs} \\
\lr{Random Forest} & \lr{n\_estimators=200} \\
\lr{HistGradientBoosting} & \lr{max\_iter=400} (فقط dense) \\
\lr{KNN} & \lr{n\_neighbors=7, metric=cosine} \\
\bottomrule
\end{tabularx}
\caption{مدل‌های پایه}
\label{tab:model_configs}
\end{table}

\subsection{نتایج کامل}

جدول زیر همه \lr{32} ترکیب (شامل سه مدل tune‌شده در انتها) رو بر اساس \lr{F1} مرتب کرده:

\begin{table}[H]
\centering
\scriptsize
\begin{tabular}{llcccc}
\toprule
\textbf{مدل} & \textbf{\lr{Vectorizer}} & \textbf{دقت} & \textbf{صحت} & \textbf{فراخوانی} & \textbf{\lr{F1}} \\
\midrule
""" + phase3_rows + r"""
\bottomrule
\end{tabular}
\caption{نتایج کامل \lr{32} آزمایش}
\label{tab:full_results}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{../outputs/figures/all_confusion_matrices.png}
    \caption{ماتریس درهم‌ریختگی همه مدل‌ها}
    \label{fig:all_cms}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{../outputs/figures/phase3_f1_comparison.png}
    \caption{مقایسه \lr{F1} روی پنج \lr{vectorizer}}
    \label{fig:f1_comparison}
\end{figure}

\subsection{تحلیل نتایج}

بهترین ترکیب بعد از \lr{GridSearchCV}: \lr{SVM (Linear) (tuned)} روی \lr{Hybrid} با \lr{F1=0.8574} و دقت \lr{0.8578}. \lr{LogisticRegression} روی \lr{Hybrid} و \lr{TF-IDF} هم نزدیک (\lr{0.8563} و \lr{0.8551}). \lr{Random Forest} دیگه برنده نیست؛ روی sparseهای پُربُعد، مدل‌های خطی با \lr{C} درست قوی‌ترن.

بدترین‌ها همچنان \lr{DecisionTree} روی \lr{Word2Vec} (\lr{F1=0.7475}) و \lr{KNN} روی sparse (\lr{0.7671} روی \lr{CountVectorizer}). \lr{KNN} توی فضای \lr{50k}-بعدی sparse گم میشه.

\subsection{تنظیم ابرپارامتر و ablation پیش‌پردازش}

\lr{GridSearchCV} با \lr{5}-fold: \lr{ComplementNB} \lr{alpha=1.0} روی \lr{CountVectorizer} (\lr{F1=0.8501})؛ \lr{SVM} \lr{C=0.05} روی \lr{Hybrid} (\lr{0.8574})؛ \lr{LogisticReg} \lr{C=0.5} (\lr{0.8566}).

روی زیرنمونه \lr{15k/5k}، حذف \lr{lemmatization} (\lr{F1=0.8572}) از pipeline فعلی (\lr{0.8512}) جلو بود، ولی pipeline فعلی رو نگه داشتیم چون ablation کوچک بود و negation/emoji-mapping عمداً برای احساس طراحی شده.

\subsection{نمونه پیش‌بینی‌های اشتباه (بهترین مدل sweep)}

\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{X c c}
\toprule
\textbf{متن} & \textbf{برچسب} & \textbf{پیش‌بینی} \\
\midrule
کیفیت خوب بود ولی نسبت به قیمت حجم غذا خوب بود & \lr{SAD} & \lr{HAPPY} \\
هسته زردآلو خیلی بد بود ولی برگه هلو خوب بود & \lr{SAD} & \lr{HAPPY} \\
سالاد سزار کاهو موند رنگ عوض شد سیب‌زمینی ساده خوب بود & \lr{HAPPY} & \lr{SAD} \\
\bottomrule
\end{tabularx}
\caption{خطاهای \lr{LogisticReg + Hybrid} — الگوی «ولی»}
\label{tab:prediction_examples}
\end{table}

پس برنده sweep اولیه \lr{LogisticReg + Hybrid} بود (\lr{0.8563})؛ بعد از tune، \lr{SVM + Hybrid} به \lr{0.8574} رسید.


\section{فاز چهارم: انتخاب بهترین \lr{Vectorizer}}

\subsection{میانگین و اوج عملکرد}

\begin{table}[H]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{\lr{Vectorizer}} & \textbf{دقت} & \textbf{صحت} & \textbf{فراخوانی} & \textbf{\lr{F1}} \\
\midrule
""" + vec_avg_rows + r"""
\bottomrule
\end{tabular}
\caption{میانگین \lr{F1} روی همه مدل‌های هر \lr{vectorizer}}
\label{tab:vec_avg}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{lc}
\toprule
\textbf{\lr{Vectorizer}} & \textbf{اوج \lr{F1}} \\
\midrule
""" + peak_rows + r"""
\bottomrule
\end{tabular}
\caption{بالاترین \lr{F1} هر نمایش (بعد از tune)}
\label{tab:peak_f1}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{../outputs/figures/avg_metrics_per_vectorizer.png}
    \caption{میانگین معیارها}
    \label{fig:vec_avg}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{../outputs/figures/f1_heatmap.png}
    \caption{هیت‌مپ \lr{F1}}
    \label{fig:f1_heatmap}
\end{figure}

\subsection{چرا \lr{Hybrid} انتخاب شد؟}

\lr{Hybrid} هم میانگین (\lr{0.8515}) هم اوج (\lr{0.8574}) رو برد. char n-gram املای غلط و پسوند فارسی رو می‌گیره؛ word TF-IDF هم عبارت‌هایی مثل \lr{NEG\_خوب} رو نگه می‌داره. بین سه روش \textbf{اجباری}، \lr{TF-IDF} برنده‌ست (\lr{0.8551} اوج)، نه \lr{CountVectorizer} (\lr{0.8501}). \lr{Hybrid} در عمل نسخه تقویت‌شده \lr{TF-IDF} با char union هست.

پس برای فاز \lr{Stacking}، \lr{Hybrid} ذخیره شد (\lr{best\_vectorizer.json}).


\section{فاز پنجم: \lr{Stacking}}

\subsection{طراحی}

چهار پایه روی \lr{Hybrid} sparse: \lr{ComplementNB}، \lr{SVM} کالیبره، \lr{LogisticRegression}، \lr{RandomForest} (\lr{n\_estimators=300}). متا: \lr{LogisticRegression}، \lr{cv=5}، \lr{stack\_method=auto}. زمان آموزش حدود \lr{683} ثانیه (\lr{11.4} دقیقه).

\subsection{نتایج}

\begin{table}[H]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{مدل} & \textbf{دقت} & \textbf{صحت} & \textbf{فراخوانی} & \textbf{\lr{F1}} \\
\midrule
\lr{Stacking} (\lr{Hybrid}) & \lr{0.8580} & \lr{0.8601} & \lr{0.8580} & \lr{0.8578} \\
\lr{SVM (tuned)} (تکی) & \lr{0.8578} & \lr{0.8620} & \lr{0.8578} & \lr{0.8574} \\
\bottomrule
\end{tabular}
\caption{\lr{Stacking} در برابر بهترین تکی}
\label{tab:stacking_results}
\end{table}

بهبود \lr{F1}: \lr{+0.0004} (\lr{0.04} نقطه درصد). کم ولی مثبت.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.65\textwidth]{../outputs/figures/stacking_vs_best.png}
    \caption{مقایسه \lr{Stacking} و بهترین تکی}
    \label{fig:stacking_vs_best}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.55\textwidth]{../outputs/confusion_matrices/Stacking_Hybrid.png}
    \caption{ماتریس درهم‌ریختگی \lr{Stacking + Hybrid}}
    \label{fig:stacking_cm}
\end{figure}

\subsection{تحلیل}

\lr{Stacking} کمی بهتر از \lr{SVM tuned} شد. بهبود کوچیکه چون پایه‌ها روی یه ماتریس مشترک آموزش دیدن و \lr{SVM} خودش از قبل خیلی قوی بود. ولی متا-مدل یاد گرفته کی به کدوم پایه اعتماد کنه.

\subsection{نمونه‌های خطا و سقف دقت}

\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{X c c}
\toprule
\textbf{متن} & \textbf{برچسب} & \textbf{پیش‌بینی} \\
\midrule
کیفیت خوب بود ولی نسبت به قیمت حجم غذا خوب بود & \lr{SAD} & \lr{HAPPY} \\
هسته زردآلو خیلی بد بود ولی برگه هلو خوب بود & \lr{SAD} & \lr{HAPPY} \\
آدرس اشتباه فرستاد ولی خود همبرگر خوب بود & \lr{SAD} & \lr{HAPPY} \\
از نظر مزه خوشمزه بود ولی حجم کم بود & \lr{SAD} & \lr{HAPPY} \\
سالاد سزار کاهو موند؛ سیب‌زمینی خوب بود & \lr{HAPPY} & \lr{SAD} \\
چیپس پنیر بدون سس تند سفارش دادم ولی تند فرستادید & \lr{HAPPY} & \lr{SAD} \\
\bottomrule
\end{tabularx}
\caption{خطاهای واقعی \lr{Stacking}}
\label{tab:stacking_examples}
\end{table}

حدود \lr{1,970} خطا از \lr{13,896} تست (\lr{85.8\%} دقت). خیلیاشون نظر مختلط یا برچسب مشکوک‌ان، نه خطای تصادفی مدل. «کیفیت عالی. اما غذا سرد شده بود» (\lr{HAPPY}) یا «خوشمزه بود ولی فوق العاده سرد» (\lr{SAD}) نمونه‌ان که انسان هم مردد میشه.

پس \lr{F1} حدود \lr{0.86} هم موفقیت pipeline کلاسیکه هم یادآوری که داده ۱۰۰٪ تمیز نیست.


\section{فاز ششم (bonus): \lr{ParsBERT}}

روی \lr{Kaggle GPU} سه مسیر تست شد (منبع: \lr{06\_parsbert.ipynb}):

\textbf{۱. Frozen \lr{ParsBERT}} — embedding \lr{768}-بعدی + سرهای کلاسیک: بهترین \lr{SVM} \lr{F1=0.8445}. \lr{Stacking} روی frozen: \lr{0.8450}. هر دو \textbf{پایین‌تر} از \lr{Stacking Hybrid} (\lr{0.8578}).

\textbf{۲. Fine-tune} روی متن خام: \lr{F1=0.8730}، دقت \lr{0.8731}. جهش \lr{+0.0152} نسبت به stacking کلاسیک.

\textbf{۳. Extended meta-ensemble} (پنج عضو + متا \lr{LogReg}، \lr{C=0.3}): \lr{F1=0.8713} — کمی پایین‌تر از fine-tune تنها؛ سود اصلی از خود \lr{ParsBERT} فاین‌تیون‌شده میاد.

\begin{table}[H]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{مرحله} & \textbf{\lr{F1}} & \textbf{نسبت به \lr{Stacking}} \\
\midrule
\lr{Stacking Hybrid} (خط مبنا) & \lr{0.8578} & \lr{0} \\
\lr{ParsBERT frozen — stacking} & \lr{0.8450} & \lr{-0.0128} \\
\lr{ParsBERT fine-tuned} & \lr{0.8730} & \lr{+0.0152} \\
\lr{Extended ensemble} & \lr{0.8713} & \lr{+0.0135} \\
\bottomrule
\end{tabular}
\caption{خلاصه bonus}
\label{tab:parsbert}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{../outputs/figures/parsbert_vs_classical.png}
    \caption{مقایسه \lr{ParsBERT} و pipeline کلاسیک}
    \label{fig:parsbert}
\end{figure}

پس برای عبور از سقف \lr{bag-of-words}، context لازمه؛ frozen کافی نیست، fine-tune لازمه.


\section{نتیجه‌گیری}

یه pipeline کامل روی \lr{69,479} نظر \lr{Snapfood}: پیش‌پردازش احساس‌محور (نفی، \lr{emoji})، پنج \lr{vectorizer}، \lr{32+} آزمایش مدل، انتخاب \lr{Hybrid}، \lr{Stacking} با \lr{F1=0.8578}.

نکات کلیدی: \lr{URL} قبل از \lr{Hazm}؛ \lr{stopword} احساسی حذف نشه؛ \lr{Hybrid} برنده sparse؛ \lr{SVM tuned} قوی‌تر از درخت‌ها؛ \lr{Stacking} بهبود کوچک ولی مثبت؛ خطاها اغلب نظر مختلط/برچسب نویزی؛ \lr{ParsBERT} فاین‌تیون‌شده به \lr{0.8730} رسید.

تحویل اجباری پروژه با فازهای ۱–۵ کامله؛ فاز ۶ bonus و جهش واقعی با ترنسفورمر فاین‌تیون‌شده بود.

\end{document}
"""

header = REPORT.read_text(encoding="utf-8").split(r"\section{معرفی داده‌ها}")[0].rstrip()
REPORT.write_text(header + body, encoding="utf-8")
print("report rewritten", len(body), "chars body")
