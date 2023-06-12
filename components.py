def card(text, label, sentiment):
    if sentiment == "Positive":
        return f"""
        <div class="card bg-dark m-4" style="width: 30rem;">
            <div class="card-body">
                <div class="flex flex-row justify-content-between">
                    <h5 class="lead">{label}</h5>
                    <span class="badge badge-success">Postive</span>
                <div>
                <p class="card-text">{text}</p>
            </div>
        </div>
        """
    elif sentiment == "Negative":
        return f"""
        <div class="card bg-dark m-4" style="width: 30rem;">
            <div class="card-body">
                <div class="flex flex-row justify-content-between">
                    <h5 class="lead">{label}</h5>
                    <span class="badge badge-danger">Negative</span>
                </div>
                <p class="card-text">{text}</p>
            </div>
        </div>
        """
    else:
        return f"""
        <div class="card bg-dark m-4" style="width: 30rem;">
            <div class="card-body">
                    <div class="flex flex-row justify-content-between">
                        <h5 class="lead">{label}</h5>
                        <span class="badge badge-warning">Neutral</span>
                    </div>
                <p class="card-text">{text}</p>
            </div>
        </div>
        """


