<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
    public function up(): void {
        Schema::create('e_wallets', function (Blueprint $table) {
            $table->id('id_wallet');
            $table->decimal('saldo', 15, 2)->default(0.00);
            $table->foreignId('id_user')->constrained('users', 'id_user')->onDelete('cascade');
            $table->timestamps();
        });
    }
    public function down(): void { Schema::dropIfExists('e_wallets'); }
};