<?php

use App\Models\FactUserDailyMetric;
use App\Models\Transaksi;
use Carbon\Carbon;

require __DIR__.'/vendor/autoload.php';
$app = require_once __DIR__.'/bootstrap/app.php';
$kernel = $app->make(Illuminate\Contracts\Console\Kernel::class);
$kernel->bootstrap();

echo "--- Debugging Data ---\n";

// 1. Check Fact Table Count
$factCount = FactUserDailyMetric::count();
echo "FactUserDailyMetric Count: $factCount\n";

if ($factCount > 0) {
    // 2. Check Sample Data
    $sample = FactUserDailyMetric::orderBy('date', 'desc')->take(5)->get();
    echo "\nSample Facts (Last 5):\n";
    foreach ($sample as $s) {
        echo "Date: {$s->date} | User: {$s->user_id} | Income: {$s->total_income}\n";
    }

    // 3. Check Date Range for Query
    $startDate = Carbon::now()->subDays(29)->toDateString();
    $endDate = Carbon::now()->toDateString();
    echo "\nQuery Range: $startDate to $endDate\n";

    $queryCount = FactUserDailyMetric::whereBetween('date', [$startDate, $endDate])->count();
    echo "Facts in Range: $queryCount\n";
    
    // 4. Check Group By logic
    $gmvTrend = FactUserDailyMetric::whereBetween('date', [$startDate, $endDate])
            ->selectRaw('date, SUM(total_income) as total_gmv')
            ->groupBy('date')
            ->orderBy('date')
            ->get();
            
    echo "\nGMV Trend Result:\n";
    foreach($gmvTrend as $trend) {
        echo "Date: {$trend->date} -> GMV: {$trend->total_gmv}\n";
    }
} else {
    echo "Fact Table is Empty! ETL failed?\n";
    // Check one transaction
    $trx = Transaksi::latest()->first();
    if ($trx) {
        echo "\nLatest Transaction: {$trx->updated_at} (ID: {$trx->id_transaksi})\n";
    }
}
