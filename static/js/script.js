$(document).ready(function() {
    // Prediction form submission
    $('#predict-form').submit(function(event) {
        event.preventDefault();
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: $(this).serialize(),
            success: function(response) {
                if (response.error) {
                    $('#prediction-result').html(`<div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md">${response.error}</div>`);
                } else {
                    $('#prediction-result').html(`
                        <div class="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 rounded-md">
                            <strong>Predicted Category:</strong> ${response.category}<br>
                            <strong>Predicted Priority:</strong> ${response.priority}<br>
                            <strong>Predicted Duration:</strong> ${response.duration}<br>
                        </div>
                    `);
                }
            },
            error: function() {
                $('#prediction-result').html('<div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md">An error occurred. Please try again.</div>');
            }
        });
    });

    // Forecast form submission
    let forecastChart = null;
    $('#forecast-form').submit(function(event) {
        event.preventDefault();
        $.ajax({
            url: '/forecast',
            type: 'POST',
            data: $(this).serialize(),
            success: function(response) {
                if (response.error) {
                    $('#forecast-result').html(`<div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md">${response.error}</div>`);
                } else {
                    // Destroy existing chart if any
                    if (forecastChart) {
                        forecastChart.destroy();
                    }
                    // Create new chart
                    const ctx = document.getElementById('forecast-chart').getContext('2d');
                    forecastChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: response.dates,
                            datasets: [
                                {
                                    label: 'Predicted Tasks',
                                    data: response.yhat,
                                    borderColor: '#2563eb',
                                    fill: false
                                },
                                {
                                    label: 'Lower Bound',
                                    data: response.yhat_lower,
                                    borderColor: '#16a34a',
                                    borderDash: [5, 5],
                                    fill: false
                                },
                                {
                                    label: 'Upper Bound',
                                    data: response.yhat_upper,
                                    borderColor: '#dc2626',
                                    borderDash: [5, 5],
                                    fill: false
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Date'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Number of Tasks'
                                    }
                                }
                            }
                        }
                    });
                }
            },
            error: function() {
                $('#forecast-result').html('<div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md">An error occurred. Please try again.</div>');
            }
        });
    });
});
