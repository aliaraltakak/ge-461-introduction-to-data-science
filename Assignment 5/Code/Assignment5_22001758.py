# Author: Ali Aral Takak / Student ID: 22001758
# Import required libraries.
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from skmultiflow.data import AGRAWALGenerator, SEAGenerator, FileStream
from skmultiflow.meta import AdaptiveRandomForestClassifier as ARF
from skmultiflow.lazy import SAMKNNClassifier as SAMkNN
from ensemblers import ActiveEnsemble, PassiveEnsemble

# Set random seed for reproducibility.
np.random.seed(22001758)

# Config variables.
streamLength = 100_000                           
segmentsCount = 20                              
dataDir = '/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 5/Datasets'  
outputDir = 'outputs'                             
os.makedirs(outputDir, exist_ok=True)

def generateStream(generator, totalSamples, driftPoints, outputPath):
    # Create parent directory if needed.
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    # Open CSV file to write generated data.
    with open(outputPath + '.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        # Write header row: feature0, feature1, ..., target.
        writer.writerow([f'feature{i}' for i in range(generator.n_features)] + ['target'])
        # Generate samples with concept drift at specified points.
        for idx in range(totalSamples):
            if idx in driftPoints:
                generator.prepare_for_use()  # Reset generator state at drift.
            xSample, ySample = generator.next_sample()  # Fetch next sample.
            # Write feature values and label to CSV.
            writer.writerow(list(xSample[0]) + [int(ySample[0])])

def evaluate(models, stream, nSegments=segmentsCount):
    # Determine total samples and segment size.
    totalSamples = stream.n_remaining_samples()
    segmentSize = totalSamples // nSegments

    # Initialize trackers for accuracy.
    overallCorrect = {name: 0 for name, _ in models}     # Total correct count per model.
    segmentAccs    = {name: [] for name, _ in models}    # Accuracy per segment.
    segmentCounts  = {name: 0 for name, _ in models}     # Samples counted in current segment.
    segmentCorrect = {name: 0 for name, _ in models}     # Correct count in current segment.

    # Bootstrap: train each model on first sample to initialize.
    initialX, initialY = stream.next_sample()
    for name, learner in models:
        try:
            learner.partial_fit(initialX, initialY, classes=[0,1]) 
        except TypeError:
            learner.partial_fit(initialX, initialY)             
    stream.restart()  # Reset stream to beginning.

    # Main prequential loop: test-then-train.
    for idx in range(totalSamples):
        xInst, yInst = stream.next_sample()
        for name, learner in models:
            rawPred = learner.predict(xInst)                       
            # Safely extract single prediction value.
            try:
                pred = rawPred[0]
            except (TypeError, IndexError):
                pred = rawPred
            correct = int(pred == yInst[0])                      

            # Update overall and segment counters.
            overallCorrect[name] += correct
            segmentCorrect[name] += correct
            segmentCounts[name]  += 1

            # Train model on current instance.
            learner.partial_fit(xInst, yInst)

        # At end of segment, compute and reset segment accuracy.
        if (idx + 1) % segmentSize == 0:
            for name, _ in models:
                acc = segmentCorrect[name] / segmentCounts[name]
                segmentAccs[name].append(acc)
                segmentCorrect[name] = 0
                segmentCounts[name]  = 0

    # Compute overall accuracy for each model.
    overallAcc = {name: overallCorrect[name] / totalSamples for name, _ in models}
    return overallAcc, segmentAccs

def runOnStream(streamCsv, streamName):
    # Print header for current stream evaluation.
    print(f"\n{streamName}:")
    stream = FileStream(streamCsv)

    # Define models to evaluate.
    models = [
        ("ARF",        ARF()),                         # Adaptive Random Forest.
        ("SAMkNN",     SAMkNN(n_neighbors=5, max_window_size=200)),  # Self-adjusting kNN.
        ("ActiveEns",  ActiveEnsemble()),              # Custom active ensemble.
        ("PassiveEns", PassiveEnsemble())              # Custom passive ensemble.
    ]

    # Perform evaluation.
    overallAcc, segmentAccs = evaluate(models, stream)

    # Print overall accuracy results.
    for name, acc in overallAcc.items():
        print(f"{name:12s} overall accuracy: {acc*100:5.2f}%")

    # Plot aggregated segment accuracies.
    plt.figure(figsize=(8,4))
    for name, _ in models:
        plt.plot(range(1, segmentsCount+1), segmentAccs[name], label=name)
    plt.xlabel("Segment")
    plt.ylabel("Accuracy")
    plt.title(streamName)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, f"{streamName}_prequential.png"))
    plt.close()

# Generate synthetic streams with drifts at samples 35_000 and 60_000.
generateStream(SEAGenerator(random_state=22001758), streamLength, [35000, 60000],
               os.path.join(dataDir, 'SEADataset'))
generateStream(AGRAWALGenerator(random_state=22001758), streamLength, [35000, 60000],
               os.path.join(dataDir, 'AGRAWALGenerator'))

# Main.
if __name__ == "__main__":
    # Evaluate on synthetic and real datasets.
    runOnStream(os.path.join(dataDir, 'SEADataset.csv'),       "SEA-Generated Data")
    runOnStream(os.path.join(dataDir, 'AGRAWALGenerator.csv'), "AGRAWAL-Generated Data")
    runOnStream(os.path.join(dataDir, 'spam.csv'),             "Spam Dataset")
    runOnStream(os.path.join(dataDir, 'elec.csv'),             "Electricity Dataset")
