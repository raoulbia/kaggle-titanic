# UNDER CONSTRUCTION
### log reg with sklearn

from sklearn.linear_model import SGDClassifier

# logreg = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
# X = np.matrix(train.ix[:, 2:])
# X = np.matrix(normalize(X, axis=0, norm='max'))
# logreg.fit(X, np.ravel(y))
# logger.info(logreg.coef_)
# print(logreg._predict_proba(X))


# p = predict(X=X, theta=logreg.coef_.T, threshold=logit_threshold)
# logger.info('Train Accuracy: {}%'.format(round(np.mean(p == 1)) * 100))
# logger.info('Expected Train Accuracy: {}%'.format(round(np.count_nonzero(y) / y.shape[0] * 100)))
