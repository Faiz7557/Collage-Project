<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\User;
use App\Models\Transaksi;
use App\Models\FactUserDailyMetric;
use Illuminate\Support\Facades\DB;
use Carbon\Carbon;

class GenerateDashboardFacts extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'dashboard:aggregate {date? : Tanggal yang ingin di-aggregate (Y-m-d), default: kemarin}';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'ETL Process: Aggregate daily transaction data into Fact Table';

    /**
     * Execute the console command.
     */
    public function handle()
    {
        $dateParam = $this->argument('date');
        
        if ($dateParam === 'all') {
            $startDate = Carbon::now()->subDays(366);
            $endDate = Carbon::now();
            
            for ($date = $startDate; $date->lte($endDate); $date->addDay()) {
                $this->processDate($date);
            }
            $this->info("Completed batch processing.");
            return;
        }

        $date = $dateParam ? Carbon::parse($dateParam) : Carbon::yesterday();
        $this->processDate($date);
    }

    private function processDate($date)
    {
        $dateStr = $date->toDateString();
        $this->info("Starting ETL process for date: $dateStr");
        
        // Use database sorting/grouping to avoid N+1 problem
        // We will build a temporary collection of stats by querying transactions once for sellers and once for buyers
        
        // 1. Sales Stats (Income, Kg Sold)
        $salesStats = Transaksi::whereDate('updated_at', $dateStr)
            ->whereIn('status_transaksi', ['disetujui', 'completed', 'confirmed'])
            ->selectRaw('
                id_penjual as user_id, 
                SUM(COALESCE(harga_akhir, harga_awalan, 0) * jumlah) as income,
                SUM(jumlah) as kg_sold,
                COUNT(*) as tx_count_sold
            ')
            ->groupBy('id_penjual')
            ->get()
            ->keyBy('user_id');

        // 2. Purchase Stats (Expense, Kg Bought)
        $purchaseStats = Transaksi::whereDate('updated_at', $dateStr)
            ->whereIn('status_transaksi', ['disetujui', 'completed', 'confirmed'])
            ->selectRaw('
                id_pembeli as user_id, 
                SUM(COALESCE(harga_akhir, harga_awalan, 0) * jumlah) as expense,
                SUM(jumlah) as kg_bought,
                COUNT(*) as tx_count_bought
            ')
            ->groupBy('id_pembeli')
            ->get()
            ->keyBy('user_id');

        // 3. Merge and Upsert
        // Get all unique user IDs involved
        $allUserIds = $salesStats->keys()->merge($purchaseStats->keys())->unique();
        
        if ($allUserIds->isEmpty()) {
            $this->info("No transactions found for $dateStr.");
            return;
        }

        $this->info("Processing " . $allUserIds->count() . " active users...");

        // Prepare data for upsert
        $upsertData = [];
        $users = User::whereIn('id_user', $allUserIds)->pluck('peran', 'id_user');

        foreach ($allUserIds as $userId) {
            $sale = $salesStats->get($userId);
            $purchase = $purchaseStats->get($userId);
            
            $upsertData[] = [
                'date' => $dateStr,
                'user_id' => $userId,
                'role' => $users[$userId] ?? 'unknown',
                'total_income' => $sale->income ?? 0,
                'total_expense' => $purchase->expense ?? 0,
                'total_kg_sold' => $sale->kg_sold ?? 0,
                'total_kg_bought' => $purchase->kg_bought ?? 0,
                'transaction_count' => ($sale->tx_count_sold ?? 0) + ($purchase->tx_count_bought ?? 0),
                'created_at' => now(), // literal now() for bulk insert
                'updated_at' => now(),
            ];
        }

        // Perform Bulk Upsert
        // chunking to prevent placeholder overflow
        foreach (array_chunk($upsertData, 500) as $chunk) {
            FactUserDailyMetric::upsert(
                $chunk, 
                ['date', 'user_id'], // Unique Key
                ['total_income', 'total_expense', 'total_kg_sold', 'total_kg_bought', 'transaction_count', 'updated_at'] // Columns to update
            );
        }
        
        $this->info("Successfully processed facts for $dateStr.");
    }
}
