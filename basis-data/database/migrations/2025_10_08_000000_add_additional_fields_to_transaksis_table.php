<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('transaksis', function (Blueprint $table) {
            $table->enum('type', ['purchase', 'sale', 'topup', 'expenditure'])->nullable()->after('status_transaksi');
            $table->text('description')->nullable()->after('type');
            $table->enum('payment_method', ['bank', 'mini_market'])->nullable()->after('description');
            $table->string('reference_code')->nullable()->after('payment_method');
            $table->foreignId('user_id')->nullable()->constrained('users', 'id_user')->after('reference_code');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('transaksis', function (Blueprint $table) {
            $table->dropForeign(['user_id']);
            $table->dropColumn(['type', 'description', 'payment_method', 'reference_code', 'user_id']);
        });
    }
};
