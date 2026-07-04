"""'How it works' revision notes: a short intuition paragraph plus exam takeaways."""

NOTES = {
    # ------------------------------------------------------------ regression
    "Linear Regression": (
        "Fits the straight line (or hyperplane) that minimizes the sum of squared "
        "errors between predictions and targets. It has a closed-form solution "
        "(the normal equations), so there is nothing iterative to tune. It is the "
        "baseline every other regressor is compared against.",
        [
            "**Bias–variance:** high bias, low variance — it cannot bend, so it underfits any nonlinear relationship.",
            "**When to use:** as a first baseline, or when the relationship is roughly linear and interpretability matters.",
            "**Pitfalls:** squared loss makes it sensitive to outliers; correlated features make coefficients unstable.",
            "**Exam point:** no regularization means coefficients can blow up with multicollinearity — motivates Ridge/Lasso.",
        ],
    ),
    "Ridge Regression": (
        "Ordinary least squares plus an L2 penalty α·Σβ² that shrinks coefficients "
        "toward zero (but never exactly to zero). Shrinking trades a little bias "
        "for a large drop in variance. Here Ridge is fitted on polynomial features "
        "of x so you can *see* the shrinkage taming a wiggly curve.",
        [
            "**α ↑ → more bias, less variance:** α→0 recovers plain least squares; huge α flattens the fit toward the mean (underfit).",
            "**Keeps all features:** coefficients shrink smoothly; use Lasso if you want exact zeros.",
            "**Best with multicollinearity:** the penalty stabilizes coefficients of correlated features.",
            "**Pitfall:** the penalty depends on feature scale — always standardize features first.",
        ],
    ),
    "Lasso Regression": (
        "Least squares plus an L1 penalty α·Σ|β|. The L1 geometry pushes some "
        "coefficients *exactly* to zero, so Lasso does feature selection while it "
        "fits. Fitted here on polynomial features so you can watch high-order "
        "terms get zeroed out as α grows.",
        [
            "**Sparsity:** produces exact zero coefficients — check the bar chart as you raise α.",
            "**α too large → underfitting:** eventually every coefficient is zero and the model predicts the mean.",
            "**Correlated features:** Lasso arbitrarily picks one of a correlated group and zeroes the rest (Ridge shares the weight).",
            "**Exam point:** L1 = diamond constraint region, corners on the axes explain the exact zeros; solved by coordinate descent, not a closed form.",
        ],
    ),
    "Polynomial Regression": (
        "Expands x into powers (x, x², …, x^d) and fits ordinary linear regression "
        "on the expanded features. The model is nonlinear in x but still *linear in "
        "the parameters*, so all linear-regression theory applies. Degree d is the "
        "single knob that walks the whole bias–variance curve.",
        [
            "**Degree = complexity dial:** d=1 underfits a curve; large d chases noise and oscillates wildly near the data edges (Runge-style wiggles).",
            "**Watch the gap:** raise the degree and watch train R² climb while test R² collapses — the classic overfitting picture.",
            "**Pitfall:** terrible extrapolation — polynomial tails explode outside the training range.",
            "**Fix for high degree:** combine with Ridge/Lasso to regularize the extra flexibility.",
        ],
    ),
    "SVR": (
        "Support Vector Regression fits a tube of width ε around the data: points "
        "inside the tube cost nothing, points outside are penalized (weighted by C). "
        "Only the points on or outside the tube — the support vectors — define the "
        "fit, and kernels let the tube bend.",
        [
            "**ε controls sparsity:** a wider tube ignores more points → fewer support vectors, smoother fit.",
            "**C ↑ → less regularization:** large C punishes tube violations hard and can overfit; small C flattens the fit.",
            "**RBF γ:** small γ = long-range influence = smooth; large γ = each point only bends the curve locally = wiggly.",
            "**Pitfall:** kernel SVR is distance-based — features must be scaled (done automatically here).",
        ],
    ),
    "KNN Regressor": (
        "Predicts a point's value as the (optionally distance-weighted) average of "
        "its k nearest training targets. There is no training phase and no global "
        "equation — the training set *is* the model. The fit is a step-like curve "
        "that follows local structure.",
        [
            "**k small → high variance:** k=1 reproduces training points exactly (jagged fit); k=n predicts the global mean (flat line).",
            "**Lazy learner:** zero training cost, but every prediction searches the whole training set.",
            "**Distance-based:** feature scaling is essential; suffers badly in high dimensions (curse of dimensionality).",
            "**weights='distance'** smooths the steps by letting closer neighbours count more.",
        ],
    ),
    "Decision Tree Regressor": (
        "Recursively splits the input space with axis-aligned thresholds that "
        "minimize the variance (MSE) inside each region, then predicts the mean "
        "target of each leaf. The result is a piecewise-constant staircase fit.",
        [
            "**Depth = complexity:** deep trees carve a step per point and memorize noise; max_depth and min_samples_leaf are the main brakes.",
            "**Cannot extrapolate:** outside the training range the prediction is just the nearest leaf's constant.",
            "**No scaling needed:** splits compare a single feature against a threshold, so units don't matter.",
            "**High variance:** small data changes give very different trees — the reason forests and boosting exist.",
        ],
    ),
    "Random Forest Regressor": (
        "Trains many deep trees, each on a bootstrap sample of the data and with a "
        "random subset of features considered at each split, then averages their "
        "predictions. Averaging decorrelated trees slashes variance while keeping "
        "the low bias of deep trees.",
        [
            "**Variance ↓, bias ≈ same:** the ensemble can't be less biased than its trees, but averaging removes their noise.",
            "**More trees never overfits more** — the test curve just flattens; the cost is only compute.",
            "**Randomness is the point:** bootstrap rows + random feature subsets make trees disagree, which is what averaging exploits.",
            "**Pitfall:** loses the interpretability of a single tree; still can't extrapolate.",
        ],
    ),
    "Gradient Boosting Regressor": (
        "Builds shallow trees *sequentially*, each one fitted to the residual "
        "errors of the ensemble so far; predictions are added up, scaled by the "
        "learning rate. Boosting mainly reduces bias, one small correction at a "
        "time.",
        [
            "**learning_rate × n_estimators trade-off:** a small rate needs more trees but usually generalizes better.",
            "**Weak learners on purpose:** depth 2–4 trees; each fixes a little of what's left.",
            "**Can overfit:** unlike a forest, too many trees keeps driving train error down while test error turns up — watch the curve below.",
            "**Pitfall:** squared loss chases outliers; sequential fitting means it also fits noise if unregularized.",
        ],
    ),
    "AdaBoost Regressor": (
        "The original boosting recipe (AdaBoost.R2): fit a weak regressor, "
        "up-weight the training points with the largest errors, fit the next "
        "regressor on the re-weighted data, and combine with a weighted median. "
        "Later learners concentrate on whatever is still predicted badly.",
        [
            "**Re-weighting, not residuals:** contrast with gradient boosting, which fits residuals directly.",
            "**Outlier sensitivity:** hard points get ever more weight, so noisy outliers can hijack training.",
            "**Base learner matters:** the max_depth of the base tree sets how 'weak' each round is.",
            "**learning_rate** shrinks each learner's vote; lower it if the model overfits.",
        ],
    ),
    "Voting Regressor": (
        "Averages the predictions of several different model families trained "
        "independently on the same data. If the members make *different* mistakes, "
        "the average is more accurate and more stable than most members.",
        [
            "**Diversity is everything:** averaging three nearly identical models buys nothing; combine models with different biases (linear + tree + KNN).",
            "**Variance reduction:** errors that are uncorrelated partially cancel in the average.",
            "**No meta-learning:** weights are fixed — contrast with stacking, which *learns* how to combine members.",
            "**Pitfall:** one very bad member drags the average down; check members individually first.",
        ],
    ),
    "Bagging Regressor": (
        "Bootstrap AGGregatING: train the same base model on many bootstrap "
        "resamples of the data and average the predictions. Each resample sees a "
        "slightly different dataset, so unstable learners (deep trees) give "
        "usefully different answers.",
        [
            "**Targets variance, not bias:** helps unstable, low-bias learners; bagging a stable model (linear regression) changes little.",
            "**max_samples** controls how different the bootstraps are — smaller samples = more diversity.",
            "**Random Forest = bagging + random feature subsets** at each split (extra decorrelation).",
            "**Bonus:** each model leaves out ~37% of rows, enabling out-of-bag error estimates without a test set.",
        ],
    ),

    # -------------------------------------------------------- classification
    "Logistic Regression": (
        "Computes a linear score w·x + b and squashes it through the sigmoid to get "
        "P(class=1). Trained by minimizing cross-entropy (log loss); the decision "
        "boundary, where the probability is 0.5, is always a straight line in the "
        "input features.",
        [
            "**C is *inverse* regularization:** small C = strong L2 penalty = smoother, more biased model; large C ≈ unregularized.",
            "**Linear boundary only:** it cannot separate moons or circles unless you add engineered/polynomial features.",
            "**Outputs probabilities,** not just labels — usually reasonably calibrated, which trees are not.",
            "**Exam point:** despite the name it is a *classifier*; loss is log loss, not squared error.",
        ],
    ),
    "KNN Classifier": (
        "Classifies a point by majority vote among its k nearest training points. "
        "No training happens at all — the data is memorized and all work is done at "
        "prediction time. The decision boundary is implicit and can be arbitrarily "
        "curvy.",
        [
            "**k=1 memorizes:** train accuracy is 100% by construction, boundary is noisy islands (overfit); large k smooths toward the majority class (underfit).",
            "**Compare the small multiples below** — the same data with tiny/medium/large k is the classic bias–variance picture.",
            "**Distance-based:** scale features; use odd k in binary problems to avoid ties.",
            "**Pitfall:** slow at prediction on big data and weak in high dimensions.",
        ],
    ),
    "SVM": (
        "Finds the separating boundary with the *maximum margin* — the widest gap "
        "to the nearest points of each class. Those nearest points are the support "
        "vectors; nothing else affects the boundary. The kernel trick computes "
        "inner products in an implicit high-dimensional space, letting a linear "
        "margin there look curved here.",
        [
            "**C = margin hardness:** small C tolerates margin violations (wide, smooth margin, more bias); large C insists on classifying everything (narrow margin, overfit risk).",
            "**RBF γ = locality:** small γ → smooth boundary; large γ → tight islands around individual points (memorization).",
            "**Only support vectors matter** (ringed in the plot) — deleting any other point changes nothing.",
            "**Pitfall:** must scale features; kernel SVMs scale poorly beyond ~10⁴–10⁵ samples.",
        ],
    ),
    "Naive Bayes (Gaussian)": (
        "Applies Bayes' rule with the 'naive' assumption that features are "
        "independent given the class, modelling each feature per class as a 1-D "
        "Gaussian. Classification picks the class with the highest posterior. "
        "Training is just computing per-class means and variances — nearly instant.",
        [
            "**Why it works anyway:** the independence assumption is almost always false, but the *argmax* over classes is often still right.",
            "**Boundary shape:** quadratic in general (equal variances → linear), so it can curve but not follow arbitrary shapes.",
            "**Strengths:** tiny training cost, works with little data, naturally handles many classes.",
            "**Pitfalls:** probability estimates are overconfident when features correlate; var_smoothing guards against zero-variance features.",
        ],
    ),
    "Decision Tree Classifier": (
        "Greedily picks the feature/threshold split that most reduces impurity "
        "(Gini or entropy), then recurses on each side. Leaves predict their "
        "majority class, so the decision boundary is a set of axis-aligned "
        "rectangles.",
        [
            "**Depth = overfitting dial:** unlimited depth reaches ~100% train accuracy by carving a box per point; watch the boundary fragment as depth grows.",
            "**Interpretable:** expand the tree diagram below — you can read the model as if/else rules.",
            "**No scaling needed** and handles mixed feature types; but boundaries are always axis-parallel (diagonal separations need many splits).",
            "**High variance:** tiny data changes reshape the whole tree — the motivation for forests.",
        ],
    ),
    "Random Forest Classifier": (
        "An ensemble of deep decision trees, each trained on a bootstrap sample "
        "with random feature subsets per split; the forest predicts by majority "
        "vote (or averaged probabilities). Decorrelated trees vote away each "
        "other's noise.",
        [
            "**Variance reduction:** individual trees overfit, the vote does not — compare the boundary with a single deep tree's.",
            "**More trees never hurts accuracy,** only compute; tune max_depth / min_samples_leaf for the real bias–variance trade.",
            "**Two sources of randomness:** bootstrap rows AND random feature subsets — the second is what distinguishes it from plain bagging.",
            "**Pitfall:** probability estimates are lumpy and the model is a black box relative to one tree.",
        ],
    ),
    "Gradient Boosting Classifier": (
        "Adds shallow trees one at a time, each fitted to the gradient of the "
        "log loss — i.e., to what the current ensemble still gets wrong. The "
        "learning rate shrinks each tree's contribution so the ensemble creeps "
        "toward the answer instead of jumping.",
        [
            "**Bias reducer:** starts simple and adds capacity tree by tree — opposite philosophy to a forest (which starts complex and averages).",
            "**Overfits with too many trees:** test error is U-shaped in n_estimators — find the turn in the curve below.",
            "**Shrinkage rule of thumb:** lower learning_rate + more trees ≥ higher learning_rate + few trees.",
            "**Sequential = slow to train** and not parallel over trees, unlike a forest.",
        ],
    ),
    "AdaBoost Classifier": (
        "Trains a weak classifier (classically a depth-1 'stump'), multiplies the "
        "weights of the points it misclassified, trains the next stump on the "
        "re-weighted data, and repeats. The final prediction is a weighted vote "
        "where more accurate stumps count more.",
        [
            "**Focus on hard points:** each round is forced to fix earlier mistakes — that's the boosting idea in its purest form.",
            "**Equivalent view:** AdaBoost minimizes exponential loss, which is why label noise and outliers hurt it badly.",
            "**Stumps are enough:** even depth-1 learners combine into complex boundaries.",
            "**Exam contrast:** AdaBoost re-weights samples; gradient boosting fits residuals/gradients. Same family, different mechanics.",
        ],
    ),
    "Bagging Classifier": (
        "Trains the same base classifier on many bootstrap resamples and takes a "
        "majority vote. Because deep trees are unstable, each resample yields a "
        "noticeably different tree, and the vote smooths out their individual "
        "quirks.",
        [
            "**Pure variance reduction:** the vote's bias equals the base learner's bias; only the noise cancels.",
            "**Works best on unstable learners** (deep trees); bagging a stable learner like logistic regression barely changes it.",
            "**max_samples** tunes bootstrap diversity; ~37% of rows are left out of each bag (basis of out-of-bag scoring).",
            "**Random Forest is bagging + per-split feature randomness** — know the difference for exams.",
        ],
    ),
    "Voting Classifier": (
        "Combines several different classifier families: 'hard' voting takes the "
        "majority label, 'soft' voting averages predicted probabilities and picks "
        "the argmax. Diverse models that err on different points correct each "
        "other.",
        [
            "**Soft usually beats hard** when members output meaningful probabilities, because confidence information is kept.",
            "**Diversity requirement:** combining three trees is pointless; combining a linear model, a tree, and KNN mixes different inductive biases.",
            "**No learned weights:** contrast with stacking, where a meta-model learns how much to trust each member.",
            "**Pitfall:** hard voting with two members can tie; use odd counts or soft voting.",
        ],
    ),
    "XGBoost": (
        "An engineered gradient boosting system: second-order (gradient + hessian) "
        "optimization of a regularized objective, plus row/column subsampling, "
        "shrinkage, and clever tree-growing. Same core idea as gradient boosting, "
        "with regularization built into the objective itself.",
        [
            "**Built-in regularization** (penalties on leaf count and leaf weights) is the headline difference from vanilla GBM.",
            "**subsample < 1** adds stochasticity that both speeds training and combats overfitting.",
            "**Same knobs as GBM:** n_estimators × learning_rate trade-off, shallow max_depth; in practice you'd add early stopping on a validation set.",
            "**Why it wins competitions:** strong defaults, regularization, speed — not a fundamentally different algorithm.",
        ],
    ),

    # ------------------------------------------------------------ clustering
    "K-Means": (
        "Picks k centroids, assigns each point to its nearest centroid, moves each "
        "centroid to the mean of its points, and repeats until stable. This "
        "monotonically decreases inertia (within-cluster sum of squared "
        "distances), converging to a *local* optimum.",
        [
            "**Assumes round, similar-size clusters:** it partitions space into Voronoi cells, so rings and moons are split incorrectly — try the Circles dataset.",
            "**k must be chosen:** use the elbow in the inertia curve (below) or silhouette score; inertia alone always decreases with k.",
            "**Init matters:** k-means++ spreads initial centroids to avoid bad local optima; scaling matters because distance is Euclidean.",
            "**Pitfall:** sensitive to outliers (means get dragged); every point is forced into a cluster — no noise concept.",
        ],
    ),
    "DBSCAN": (
        "Density-based clustering: a point with at least min_samples neighbours "
        "within radius eps is a *core* point; clusters grow by connecting core "
        "points that reach each other, and points reachable from no core point are "
        "labelled noise (−1).",
        [
            "**Finds arbitrary shapes:** rings and moons come out perfectly when eps is right — the case K-Means cannot handle.",
            "**No k needed,** but eps effectively replaces it: too small → everything is noise; too big → clusters merge into one.",
            "**Has a noise concept:** grey × points below are outliers, something K-Means cannot express.",
            "**Pitfalls:** one global eps fails when cluster densities differ; distance-based, so scale features and beware high dimensions.",
        ],
    ),
    "Agglomerative Clustering": (
        "Starts with every point as its own cluster and repeatedly merges the two "
        "closest clusters until k remain. 'Closest' depends on the linkage rule, "
        "and the sequence of merges forms a tree (dendrogram) you can cut at any "
        "level.",
        [
            "**Linkage defines the geometry:** ward minimizes variance (compact, round clusters, like K-Means); single linkage chains through nearest points (follows shapes, but noise builds bridges); complete/average sit in between.",
            "**The dendrogram is the real output** — the merge heights show how separated clusters are; a long vertical gap suggests a natural k.",
            "**Deterministic:** no random init, unlike K-Means.",
            "**Pitfall:** O(n²) memory/time — fine here, infeasible for very large datasets.",
        ],
    ),

    # -------------------------------------------------- dimensionality reduction
    "PCA": (
        "Finds orthogonal directions (principal components) along which the data "
        "varies most; they are the eigenvectors of the covariance matrix, ordered "
        "by eigenvalue (variance explained). Projecting onto the top components "
        "compresses the data while losing as little variance as possible.",
        [
            "**Standardize first:** otherwise the feature with the biggest units dominates PC1 (done automatically here).",
            "**Reading the scree plot:** keep components up to the 'elbow', or enough for ~90–95% cumulative variance.",
            "**Unsupervised:** PCA never sees the labels — high variance is not the same as high class-separability, so a 2-D projection *may* mix classes.",
            "**Limits:** linear only (rotates/projects, cannot unroll curved manifolds); components are combinations of all features, hurting interpretability.",
        ],
    ),
}


# ElasticNet slots between Lasso and Polynomial in the registry
NOTES["ElasticNet Regression"] = (
    "Combines the Ridge (L2) and Lasso (L1) penalties in one objective, mixed by "
    "l1_ratio. It keeps Lasso's ability to zero-out coefficients while borrowing "
    "Ridge's stability with correlated features. Sliding l1_ratio from 0 to 1 "
    "morphs the model continuously from Ridge into Lasso.",
    [
        "**Two knobs:** alpha sets the *total* penalty strength; l1_ratio splits it between L1 (sparsity) and L2 (shrinkage).",
        "**Why not just Lasso?** With groups of correlated features Lasso keeps one arbitrarily; ElasticNet tends to keep the group together.",
        "**l1_ratio = 0 is Ridge, = 1 is Lasso** — watch the coefficient bars change character as you slide it.",
        "**Pitfall:** two hyperparameters to tune instead of one — in practice, cross-validate over a grid of both.",
    ],
)

# Key equation (LaTeX) shown in the "How it works" panel
EQUATIONS = {
    "Linear Regression": r"\min_{\beta}\ \sum_i \left(y_i - \beta^\top x_i\right)^2",
    "Ridge Regression": r"\min_{\beta}\ \sum_i (y_i - \beta^\top x_i)^2 + \alpha\lVert\beta\rVert_2^2",
    "Lasso Regression": r"\min_{\beta}\ \sum_i (y_i - \beta^\top x_i)^2 + \alpha\lVert\beta\rVert_1",
    "ElasticNet Regression": r"\min_{\beta}\ \tfrac{1}{2n}\lVert y - X\beta\rVert^2 + \alpha\left(\rho\lVert\beta\rVert_1 + \tfrac{1-\rho}{2}\lVert\beta\rVert_2^2\right)",
    "Polynomial Regression": r"\hat y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d",
    "SVR": r"\min_{w}\ \tfrac{1}{2}\lVert w\rVert^2 + C\sum_i \max\!\big(0,\ |y_i - f(x_i)| - \varepsilon\big)",
    "KNN Regressor": r"\hat y(x) = \frac{1}{k}\sum_{i \in N_k(x)} y_i",
    "Decision Tree Regressor": r"\text{split to minimize}\ \sum_{\text{leaves}} \sum_{i \in \text{leaf}} (y_i - \bar y_{\text{leaf}})^2",
    "Random Forest Regressor": r"\hat y(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)",
    "Gradient Boosting Regressor": r"F_m(x) = F_{m-1}(x) + \nu\, h_m(x),\quad h_m \approx -\nabla_{F} L",
    "AdaBoost Regressor": r"w_i \leftarrow w_i \cdot \beta^{\,1 - L_i}\quad(\text{large-error points keep more weight})",
    "Voting Regressor": r"\hat y(x) = \frac{1}{M}\sum_{m=1}^{M} \hat y_m(x)",
    "Bagging Regressor": r"\hat y(x) = \frac{1}{B}\sum_{b=1}^{B} \hat f^{*b}(x)\quad(\hat f^{*b}\ \text{fit on bootstrap } b)",
    "Logistic Regression": r"P(y{=}1\mid x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}",
    "KNN Classifier": r"\hat y(x) = \operatorname{mode}\{\,y_i : i \in N_k(x)\,\}",
    "SVM": r"\min_{w}\ \tfrac{1}{2}\lVert w\rVert^2 + C\sum_i \xi_i\ \ \text{s.t.}\ \ y_i(w^\top\phi(x_i)+b) \ge 1 - \xi_i",
    "Naive Bayes (Gaussian)": r"P(c \mid x) \propto P(c)\prod_j \mathcal{N}\!\left(x_j;\ \mu_{cj},\ \sigma_{cj}^2\right)",
    "Decision Tree Classifier": r"\text{Gini}(t) = 1 - \sum_c p_c^2\quad(\text{pick the split that reduces it most})",
    "Random Forest Classifier": r"\hat y(x) = \operatorname{majority}\{T_1(x), \dots, T_B(x)\}",
    "Gradient Boosting Classifier": r"F_m(x) = F_{m-1}(x) + \nu\, h_m(x),\quad h_m \approx -\nabla_F \log\text{-loss}",
    "AdaBoost Classifier": r"\alpha_t = \tfrac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t},\qquad H(x) = \operatorname{sign}\!\Big(\sum_t \alpha_t h_t(x)\Big)",
    "Bagging Classifier": r"\hat y(x) = \operatorname{majority}\{\hat f^{*1}(x), \dots, \hat f^{*B}(x)\}",
    "Voting Classifier": r"\text{hard: } \operatorname{majority}\ \big/\ \text{soft: } \arg\max_c \tfrac{1}{M}\sum_m P_m(c \mid x)",
    "XGBoost": r"\text{obj} = \sum_i l(y_i, \hat y_i) + \sum_t \Big(\gamma T_t + \tfrac{\lambda}{2}\lVert w_t\rVert^2\Big)",
    "K-Means": r"\min_{\{\mu_c\}}\ \sum_{c=1}^{k} \sum_{x \in C_c} \lVert x - \mu_c\rVert^2 \quad (\text{inertia})",
    "DBSCAN": r"x \text{ is a core point} \iff \big|\{\,x' : \lVert x' - x\rVert \le \varepsilon\,\}\big| \ge \text{min\_samples}",
    "Agglomerative Clustering": r"\text{merge } \arg\min_{A,B}\ d_{\text{linkage}}(A, B)\ \text{ until } k \text{ clusters remain}",
    "PCA": r"w_1 = \arg\max_{\lVert w\rVert = 1} \operatorname{Var}(Xw)\quad(\text{eigenvectors of the covariance matrix})",
}

# One-line mental picture for each algorithm
ANALOGIES = {
    "Linear Regression": "Stretch one straight ruler through the cloud of points so total squared miss is smallest.",
    "Ridge Regression": "A rubber band pulls every coefficient toward zero — the fit relaxes but never lets any snap to exactly zero.",
    "Lasso Regression": "A budget on the sum of |coefficients|: unimportant features get their budget cut to exactly zero.",
    "ElasticNet Regression": "A dial that morphs the penalty smoothly between Ridge's rubber band and Lasso's hard budget.",
    "Polynomial Regression": "Give the ruler joints: each extra degree adds a joint, letting it bend — and eventually wobble.",
    "SVR": "Fit a hose of width ε through the points; only points that poke outside the hose shape it.",
    "KNN Regressor": "Ask the k most similar past examples and report their average answer.",
    "Decision Tree Regressor": "Play 20-questions on the features, then answer with the average of whoever's left in the room.",
    "Random Forest Regressor": "Ask a crowd of decorrelated 20-questions players and average their answers — the noise cancels.",
    "Gradient Boosting Regressor": "A committee where each new member's only job is to fix what the committee still gets wrong.",
    "AdaBoost Regressor": "After each round, the worst-predicted points shout louder for the next learner's attention.",
    "Voting Regressor": "Average the answers of a linear thinker, a tree thinker, and a nearest-neighbour thinker.",
    "Bagging Regressor": "Train the same model on many reshuffled samples of the data, then average away their disagreements.",
    "Logistic Regression": "Draw one straight line, then grade each point's distance from it as a probability via the S-curve.",
    "KNN Classifier": "You are the majority vote of your k nearest neighbours.",
    "SVM": "Push the widest possible street between the classes; only the points touching the curbs matter.",
    "Naive Bayes (Gaussian)": "Each class is a bell curve per feature; a new point joins whichever set of bells rings loudest.",
    "Decision Tree Classifier": "A flowchart of yes/no questions that carves the space into labelled rectangles.",
    "Random Forest Classifier": "Hundreds of flowcharts, each seeing slightly different data, settle it by vote.",
    "Gradient Boosting Classifier": "Stack tiny correction trees, each nudging the ensemble toward the points it still misclassifies.",
    "AdaBoost Classifier": "Each round re-weights the mistakes so the next simple rule is forced to face them.",
    "Bagging Classifier": "Clone the classifier on bootstrap samples and let the clones out-vote each other's quirks.",
    "Voting Classifier": "A panel of different experts; take the majority (hard) or average their confidence (soft).",
    "XGBoost": "Gradient boosting with a built-in accountant that fines every extra leaf and large weight.",
    "K-Means": "Drop k magnets in the cloud; points cling to the nearest magnet, magnets slide to their crowd's centre, repeat.",
    "DBSCAN": "Clusters are crowds: walk from dense point to dense point; whoever no crowd can reach is noise.",
    "Agglomerative Clustering": "Everyone starts alone; repeatedly merge the two closest groups and record the family tree.",
    "PCA": "Rotate the axes to look at the cloud from the angle where it appears widest, then flattest, and so on.",
}
