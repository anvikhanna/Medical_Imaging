from sklearn.metrics import classification_report,confusion_matrix



Y_pred = model.predict_classes(X_test)
#returns classes

print(Y_pred)

target_names = ['class 0(Normal)', 'class 1(Abnormal)']
print(classification_report(np.argmax(Y_test, axis=1), y_pred,target_names=target_names))
#classification returns precision,recall,f1 score,support

print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))
