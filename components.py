def card(text, label, sentiment):
    if sentiment == "Positive":
        return f"""
        <div class="card bg-dark m-4" style="width: 18rem;">
            <div class="card-body">
                    <h5 class="lead">{label}</h5>
                <span class="badge badge-success">Postive</span>
                <p class="card-text">{text}</p>
            </div>
        </div>
        """
    elif sentiment == "Negative":
        return f"""
        <div class="card bg-dark m-4" style="width: 18rem;">
            <div class="card-body">
                <h5 class="lead">{label}</h5>
                <span class="badge badge-danger">Negative</span>
                <p class="card-text">{text}</p>
            </div>
        </div>
        """
    else:
        return f"""
        <div class="card bg-dark m-4" style="width: 18rem;">
            <div class="card-body">
                <h5 class="lead">{label}</h5>
                <span class="badge badge-warning">Neutral</span>
                <p class="card-text">{text}</p>
            </div>
        </div>
        """


