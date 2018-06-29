############ use test data

# test=pd.read_csv("../local-data/test-clean.csv", sep=",", header=0, index_col=0)
# # print(test.head())
#
# X_ids = test.index.values
# # print(X_ids)
#
# X = np.matrix(normalize(test)) # IMPORTANT: normalize!
# # print(X[:5,:])
# print('shape X', X.shape)
#
#
# # add intercept terms
# a = np.ones((X.shape[0], 1))
# X = np.append(a, X, axis=1)
# # print(X[:5,:])
# print('shape X after adding ones', X.shape)
#
# p = predict(X=X, theta=theta_optimized)
# # print(type(pd.DataFrame(p)))
# # print('Predicted % Survived:', round(np.mean(p == 1),2) * 100)
#
# res = pd.concat([pd.DataFrame(X_ids), pd.DataFrame(p)], axis=1)
# # res.to_csv("../local-data/result.csv", index=False, header=False)
# res.to_csv("../local-data/result_reg.csv", index=False, header=False)
