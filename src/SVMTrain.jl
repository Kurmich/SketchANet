#Linear SVM model trained using penultimate layer
module SVMTrain
import SVR
using JLD

function getaccuracy(svmmodel, features, ygold)
  ypred = SVR.predict(svmmodel, features)
  count = 0.0
  for i=1:length(ypred)
    if ypred[i] == ygold[i]
      count += 1
    end
  end
  return count/length(ypred)
end

function main()
    trnfeats = load("/home/kurmanbek/Desktop/SketchANet/svmfeatures/trnfeats.jld")["feats"]
    featmax = maximum(trnfeats)
    trnlabels = load("/home/kurmanbek/Desktop/SketchANet/svmfeatures/trnlabels.jld")["labels"]
    tstfeats = load("/home/kurmanbek/Desktop/SketchANet/svmfeatures/tstfeats.jld")["feats"]
    featmax = max(featmax, maximum(tstfeats))
    tstlabels = load("/home/kurmanbek/Desktop/SketchANet/svmfeatures/tstlabels.jld")["labels"]
    trnfeats = trnfeats ./ featmax
    tstfeats = tstfeats ./ featmax
    println("Trn data $(size(trnfeats))")
    CS = [1.0 2.0 4.0 8.0 16.0 1e2 5e2 1e3]
    tstacc = 0
    for C in CS
      svmmodel = SVR.train(trnlabels, trnfeats; svm_type=SVR.C_SVC, kernel_type=SVR.LINEAR, C=C)
      acc = getaccuracy(svmmodel, tstfeats, tstlabels)
      tstacc = max(acc, tstacc)
      println("C=$(C), tst acc=$(tstacc)")
      SVR.freemodel(svmmodel)
    end
end
main()
end
