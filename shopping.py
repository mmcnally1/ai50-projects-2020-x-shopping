import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """ 
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    Admin = []
    Admin_Duration = []
    Info = []
    Info_Duration = []
    Product_Related = []
    PR_Duration = []
    Bounce_Rates = []
    Exit_Rates = []
    Page_Values = []
    Special_Day = []
    Month = []
    Operating_Systems = []
    Browser = []
    Region = []
    Traffic_Type = []
    Visitor_Type = []
    Weekend = []
    labels = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            a = int(row["Administrative"])
            Admin.append(a)
            b = float(row["Administrative_Duration"])
            Admin_Duration.append(b)
            c = int(row["Informational"])
            Info.append(c)
            d = float(row["Informational_Duration"])
            Info_Duration.append(d)
            e = int(row["ProductRelated"])
            Product_Related.append(e)
            f = float(row["ProductRelated_Duration"])
            PR_Duration.append(f)
            g = float(row["BounceRates"])
            Bounce_Rates.append(g)
            h = float(row["ExitRates"])
            Exit_Rates.append(h)
            i = float(row["PageValues"])
            Page_Values.append(i)
            j = float(row["SpecialDay"])
            Special_Day.append(j)
            k = row["Month"]
            Month.append(k)
            l = int(row["OperatingSystems"])
            Operating_Systems.append(l)
            m = int(row["Browser"])
            Browser.append(m)
            n = int(row["Region"])
            Region.append(n)
            o = int(row["TrafficType"])
            Traffic_Type.append(o)
            p = row["VisitorType"]
            Visitor_Type.append(p)
            q = row["Weekend"]
            Weekend.append(q)
            r = row["Revenue"]
            labels.append(r)
    
    for i in range(len(Month)):
        if Month[i] == "Jan":
            Month[i] = 0
        if Month[i] == "Feb":
            Month[i] = 1
        if Month[i] == "Mar":
            Month[i] = 2
        if Month[i] == "April" or Month[i] == "Apr":
            Month[i] = 3
        if Month[i] == 'May':
            Month[i] = 4
        if Month[i] == "June" or Month[i] == "Jun":
            Month[i] = 5
        if Month[i] == "July" or Month[i] == "Jul":
            Month[i] = 6
        if Month[i] == "Aug":
            Month[i] = 7
        if Month[i] == "Sep":
            Month[i] = 8
        if Month[i] == "Oct":
            Month[i] = 9
        if Month[i] == "Nov":
            Month[i] = 10
        if Month[i] == "Dec":
            Month[i] = 11
    
    for i in range(len(Visitor_Type)):
        if Visitor_Type[i] == "Returning_Visitor":
            Visitor_Type[i] = 1
        else:
            Visitor_Type[i] = 0

    for i in range(len(Weekend)):
        if Weekend[i] == "TRUE":
            Weekend[i] = 1
        if Weekend[i] == "FALSE":
            Weekend[i] = 0
    
    for i in range(len(labels)):
        if labels[i] == "TRUE":
            labels[i] = 1
        if labels[i] == "FALSE":
            labels[i] = 0
    
    user = []
    evidence = []
    for i in range(len(Admin)):
        a = Admin[i]
        b = Admin_Duration[i]
        c = Info[i]
        d = Info_Duration[i]
        e = Product_Related[i]
        f = PR_Duration[i]
        g = Bounce_Rates[i]
        h = Exit_Rates[i]
        j = Page_Values[i]
        k = Special_Day[i]
        l = Month[i]
        m = Operating_Systems[i]
        n = Browser[i]
        o = Region[i]
        p = Traffic_Type[i]
        q = Visitor_Type[i]
        r = Weekend[i]
        s = (a,b,c,d,e,f,g,h,j,k,l,m,n,o,p,q,r)
        evidence.append(s)

    return evidence, labels

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neigh = KNeighborsClassifier(n_neighbors=1)
    model = neigh.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    correct_pos = 0
    total_pos = 0
    correct_neg = 0
    total_neg = 0

    for i in range(len(labels)):
        if predictions[i] == 1 and labels[i] == 1:
            correct_pos += 1
            total_pos += 1
        if predictions[i] == 1 and labels[i] == 0:
            total_neg += 1
        if predictions[i] == 0 and labels[i] == 0:
            correct_neg += 1
            total_neg += 1
        if predictions[i] == 0 and labels[i] == 1:
            total_pos += 1

    sensitivity = correct_pos / total_pos
    specificity = correct_neg / total_neg
        
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
